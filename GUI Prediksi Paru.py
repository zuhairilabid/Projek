import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import io
import json
import numpy as np
import base64
import datetime

# --- 1. Model & Data Libraries ---
import tensorflow as tf
from tensorflow.keras.models import load_model 

# --- 2. Cryptography Library (PyCryptodome) ---
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256, HMAC
from Crypto.Cipher import AES
from Crypto.Util import Counter
from Crypto import Random
from Crypto.Signature import pkcs1_15 # Not strictly needed, but imported in the original user code's crypto block. Keeping for context.

# --- 3. Database Library ---
from pymongo import MongoClient
#from pymongo.errors import ConnectionError as MongoConnectionError, PyMongoError
import gridfs
from bson import ObjectId

# --- CONFIGURATION ---

# The expected size for model input
IMG_SIZE = 256 
# Disease labels for classification
DISEASE_LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural_Effusion']
# MongoDB Connection String (Replace with your actual string)
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DATABASE = "CheXpert-DB"
MONGO_COLLECTION = "encrypted_results"
# Grab Model From MongoDB 
def read_model_from_mongodb(file_id, mongo_uri, db_name):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db)

    file_obj = fs.get(file_id)
    data = file_obj.read()
    filename = file_obj.filename
    with open("Densenet121_model.keras", "wb") as f:
        f.write(data)
    return "Densenet121_model.keras"
file_id = ObjectId("69393e5703a4572677d64faf")
MODEL_PATH = read_model_from_mongodb(file_id, MONGO_URI, MONGO_DATABASE)


# Pad color for resizing (e.g., black)
PAD_COLOR = (0, 0, 0)

# --- CORE CRYPTOGRAPHY FUNCTIONS (PyCryptodome) ---

def generate_key_from_password(password: str) -> tuple[bytes, bytes]:
    """
    Generates a secure 32-byte AES key from a user password using PBKDF2 with SHA256.
    A new random salt is generated for each encryption.
    Returns: (key_bytes, salt_bytes)
    """
    password_bytes = password.encode()
    # Generate a unique salt for this encryption (16 bytes recommended)
    salt = Random.get_random_bytes(16) 
    
    key = PBKDF2(
        password=password_bytes,
        salt=salt,
        dkLen=32,
        count=480000,
        # The prf function used in the original code for key derivation
        prf=lambda p, s: HMAC.new(p, s, SHA256).digest()
    )

    return key, salt

def get_key_from_password_and_salt(password: str, salt: bytes) -> bytes:
    """
    Generates the AES key from a password and a *known* salt for decryption.
    """
    password_bytes = password.encode()
    
    key = PBKDF2(
        password=password_bytes,
        salt=salt,
        dkLen=32,
        count=480000,
        # Must use the exact same PRF (Pseudo-Random Function) as encryption
        prf=lambda p, s: HMAC.new(p, s, SHA256).digest() 
    )
    return key

def encrypt_data(data_to_encrypt: bytes, key: bytes, salt: bytes) -> bytes:
    """
    Encrypts data using AES-256 in GCM mode (Authenticated Encryption)
    and prepends the salt and IV to the ciphertext.
    Returns: combined_encrypted_data (salt + iv + ciphertext + tag)
    """
    # Generate a random 16-byte IV (Nonce) for GCM mode
    nonce = Random.get_random_bytes(16)
    
    # Pad data to be a multiple of the AES block size (not strictly required for GCM, but good practice)
    # GCM is a streaming cipher, so padding is not necessary for the cipher itself.
    
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    
    ciphertext, tag = cipher.encrypt_and_digest(data_to_encrypt)
    
    # Combine salt, nonce, ciphertext, and authentication tag for storage/transmission
    # This structure is necessary for decryption
    encrypted_payload = salt + nonce + ciphertext + tag
    return encrypted_payload

def decrypt_data(encrypted_data: bytes, password: str) -> bytes:
    """
    Decrypts the combined encrypted data payload (salt + nonce + ciphertext + tag) 
    using the provided password and AES-256 GCM.
    
    Raises: ValueError on InvalidTag (authentication failure) or incorrect payload size.
    Returns: decrypted_data (JSON bytes)
    """
    SALT_SIZE = 16
    NONCE_SIZE = 16
    TAG_SIZE = 16
    
    if len(encrypted_data) < SALT_SIZE + NONCE_SIZE + TAG_SIZE:
        raise ValueError("Encrypted data payload is too short.")

    # 1. Separate components from the payload
    salt = encrypted_data[:SALT_SIZE]
    nonce = encrypted_data[SALT_SIZE:SALT_SIZE + NONCE_SIZE]
    ciphertext = encrypted_data[SALT_SIZE + NONCE_SIZE:-TAG_SIZE]
    tag = encrypted_data[-TAG_SIZE:]
    
    # 2. Re-derive the key using the password and the stored salt
    key = get_key_from_password_and_salt(password, salt)
    
    # 3. Create the AES cipher object
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    
    # 4. Decrypt and verify the authentication tag
    try:
        decrypted_data = cipher.decrypt_and_verify(ciphertext, tag)
        return decrypted_data
    except ValueError:
        raise ValueError("Decryption failed. Authentication tag is invalid. Check your password.")


# --- IMAGE PROCESSING FUNCTION (Pillow) ---

def resize_and_pad_image(image_path: str, target_size: int, pad_color: tuple) -> Image:
    """
    Loads an image and resizes it to the target_size x target_size 
    while maintaining aspect ratio by padding with the specified color.
    """
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    target_wh = target_size
    
    ratio = min(target_wh / width, target_wh / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    new_img = Image.new("RGB", (target_wh, target_wh), pad_color)
    
    left = (target_wh - new_width) // 2
    top = (target_wh - new_height) // 2
    
    new_img.paste(img, (left, top))
    
    return new_img


class DensenetClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("Densenet Chest X-ray Classifier (PyCryptodome)")

        # --- State Variables ---
        self.model = None
        self.uploaded_image_path = None
        self.current_image_tk = None 
        self.processed_image_array = None 
        self.last_prediction = None 
        self.last_encrypted_data = None 
        # New: Store the salt, as it's required for decryption
        self.last_salt = None 

        self._create_widgets()
        self._load_model()

    def _load_model(self):
        """Attempts to load the Keras model."""
        try:
            self.model = load_model(MODEL_PATH)
            self.model_status_label.config(text="Model Loaded Successfully!", fg="green")
        except Exception as e:
            self.model_status_label.config(text=f"ERROR: Model load failed: {e}", fg="red")
            messagebox.showerror("Model Error", f"Could not load model from {MODEL_PATH}.\nError: {e}")
            self.master.quit() 

    def _create_widgets(self):
        """Creates all GUI elements."""
        
        control_frame = tk.Frame(self.master, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(control_frame, text="*** CHEST X-RAY CLASSIFIER ***", font=('Arial', 12, 'bold')).pack(pady=10)
        
        self.model_status_label = tk.Label(control_frame, text="Loading Model...", fg="orange")
        self.model_status_label.pack(pady=5)
        
        tk.Button(control_frame, text="1. Upload & Preprocess X-ray", command=self.upload_image).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="2. Classify Image", command=self.classify_image, bg="#4CAF50", fg="white").pack(fill=tk.X, pady=10)
        tk.Button(control_frame, text="3. Encrypt & Save Results", command=self.encrypt_and_save, bg="#2196F3", fg="white").pack(fill=tk.X, pady=10)
        tk.Button(control_frame, text="4. Upload Encrypted to MongoDB", command=self.upload_to_mongodb, bg="#FF9800", fg="black").pack(fill=tk.X, pady=10)
        
        # New Decryption Button
        tk.Button(control_frame, text="5. Decrypt Saved Results", command=self.decrypt_saved_file, bg="#9C27B0", fg="white").pack(fill=tk.X, pady=10)

        display_frame = tk.Frame(self.master, padx=10, pady=10)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(display_frame, text=f"Processed Image ({IMG_SIZE}x{IMG_SIZE})", font=('Arial', 10, 'bold')).pack()
        
        self.image_label = tk.Label(display_frame, borderwidth=2, relief="groove", width=300, height=300)
        self.image_label.pack(pady=10)

        tk.Label(display_frame, text="Classification Results", font=('Arial', 10, 'bold')).pack()
        
        self.results_text = tk.Text(display_frame, height=10, width=50, state=tk.DISABLED)
        self.results_text.pack(pady=10)

    # --- Step 1: Upload & Preprocess Image ---
    def upload_image(self):
        """Uploads, resizes, and prepares the image array for the model."""
        f_path = filedialog.askopenfilename(
            title="Select Chest X-ray Image",
            filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*"))
        )
        if not f_path:
            return

        self.uploaded_image_path = f_path
        
        try:
            # 1. Resize and Pad Image in memory
            # 
            processed_img_pil = resize_and_pad_image(f_path, IMG_SIZE, PAD_COLOR)

            # 2. Convert for Model Input
            img_array = tf.keras.utils.img_to_array(processed_img_pil)
            img_array = np.expand_dims(img_array, axis=0) 
            self.processed_image_array = img_array / 255.0 # Normalization

            # 3. Display the resized image
            display_img = processed_img_pil.resize((300, 300), Image.LANCZOS)
            self.current_image_tk = ImageTk.PhotoImage(display_img)
            self.image_label.config(image=self.current_image_tk)
            self.image_label.image = self.current_image_tk 

            self._update_results_text(f"Image loaded and resized to {IMG_SIZE}x{IMG_SIZE}.\nReady for classification.")
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not load, resize, or display image.\nError: {e}")
            self.uploaded_image_path = None
            self.processed_image_array = None

    # --- Step 2: Classification ---
    def classify_image(self):
        """Runs the preprocessed image through the Densenet model."""
        if not self.model:
            messagebox.showerror("Error", "Model is not loaded. Cannot classify.")
            return
        if self.processed_image_array is None:
            messagebox.showerror("Error", "Please upload and preprocess an image first.")
            return

        try:
            processed_img = self.processed_image_array
            
            # Make Prediction
            predictions = self.model.predict(processed_img)
            
            # Interpret results
            results = {}
            output_text = "Classification Complete:\n\n"
            for i, label in enumerate(DISEASE_LABELS):
                certainty = predictions[0][i] * 100 
                results[label] = float(certainty)
                output_text += f"**{label}**: {certainty:.2f}%\n"
            
            self._update_results_text(output_text)
            self.last_prediction = results
            messagebox.showinfo("Success", "Classification complete. Results displayed.")

        except Exception as e:
            self._update_results_text(f"Classification ERROR: {e}")
            messagebox.showerror("Classification Error", f"An error occurred during classification.\nError: {e}")
            self.last_prediction = None
            
    # --- Step 3: Encryption and Local Save ---
    def encrypt_and_save(self):
        """Encrypts the image and results, then prompts the user to save the file."""
        if not self.uploaded_image_path or not self.last_prediction:
            messagebox.showerror("Error", "Please classify an image first.")
            return

        # 1. Get User Password
        password = simpledialog.askstring("Encryption Key", "Enter a password (used with SHA256) to encrypt results:", show='*')
        if not password:
            return

        try:
            # 2. Generate Key and Salt (using PBKDF2 with SHA256)
            key, salt = generate_key_from_password(password)

            # 3. Read original image as bytes
            with open(self.uploaded_image_path, 'rb') as f:
                image_bytes = f.read()

            # 4. Prepare JSON payload
            payload = {
                'filename': self.uploaded_image_path.split('/')[-1],
                'classification_results': self.last_prediction,
                # Store the original image bytes, base64 encoded
                'image_data': base64.b64encode(image_bytes).decode('utf-8') 
            }
            json_data = json.dumps(payload).encode('utf-8')

            # 5. Encrypt the JSON data using AES-256 GCM
            # The salt is automatically included in the encrypted_data by the encrypt_data function
            encrypted_data = encrypt_data(json_data, key, salt)
            
            self.last_encrypted_data = encrypted_data
            self.last_salt = salt # This is the *new* salt generated for this specific encryption
            
            # 6. Prompt to save the encrypted file
            save_path = filedialog.asksaveasfilename(
                defaultextension=".enc",
                filetypes=(("Encrypted File", "*.enc"), ("All files", "*.*")),
                title="Save Encrypted Results File"
            )
            
            if save_path:
                with open(save_path, 'wb') as f:
                    f.write(encrypted_data)
                
                self._update_results_text(self.results_text.get("1.0", tk.END).strip() + 
                                         f"\n\n**Data successfully encrypted (using AES-256/SHA256) and saved to:**\n{save_path}")
                messagebox.showinfo("Success", f"Encrypted file saved successfully to:\n{save_path}")
            
        except Exception as e:
            messagebox.showerror("Encryption Error", f"An error occurred during encryption/save.\nError: {e}")

    # --- Step 4: MongoDB Upload ---
    def upload_to_mongodb(self):
        """Uploads the last encrypted result to a MongoDB database."""
        if not self.last_encrypted_data:
            messagebox.showerror("Error", "No encrypted data available. Please encrypt and save first.")
            return

        if not messagebox.askyesno("Confirm Upload", "Do you want to upload the encrypted results to the MongoDB database?"):
            return

        try:
            client = MongoClient(MONGO_URI)
            db = client[MONGO_DATABASE]
            collection = db[MONGO_COLLECTION]

            # Store the encrypted payload (which includes salt, IV, ciphertext, and tag)
            mongo_document = {
                "timestamp": datetime.datetime.now(),
                "encrypted_data": self.last_encrypted_data, 
                "original_filename": self.uploaded_image_path.split('/')[-1]
            }

            result = collection.insert_one(mongo_document)
            client.close()
            
            self._update_results_text(self.results_text.get("1.0", tk.END).strip() + 
                                     f"\n\n**Successfully uploaded to MongoDB!**\nID: {result.inserted_id}")
            messagebox.showinfo("MongoDB Success", f"Encrypted results uploaded to MongoDB.\nID: {result.inserted_id}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred during MongoDB upload: {e}")
            
    # --- Step 5: Decryption of Saved File ---
    def decrypt_saved_file(self):
        """Loads an encrypted file, prompts for a password, and decrypts the data."""
        
        # 1. Select the encrypted file
        f_path = filedialog.askopenfilename(
            title="Select Encrypted Results File (.enc)",
            filetypes=(("Encrypted File", "*.enc"), ("All files", "*.*"))
        )
        if not f_path:
            return

        # 2. Get User Password
        password = simpledialog.askstring("Decryption Key", f"Enter the password for '{f_path.split('/')[-1]}':", show='*')
        if not password:
            return
            
        try:
            # 3. Read encrypted data from file
            with open(f_path, 'rb') as f:
                encrypted_data = f.read()

            # 4. Decrypt the data
            decrypted_json_bytes = decrypt_data(encrypted_data, password)
            
            # 5. Decode and parse the JSON
            decrypted_json_str = decrypted_json_bytes.decode('utf-8')
            decrypted_payload = json.loads(decrypted_json_str)

            # 6. Prepare output for display
            results = decrypted_payload.get('classification_results', {})
            filename = decrypted_payload.get('filename', 'Unknown File')

            output_text = f"*** Decryption Successful ***\n"
            output_text += f"Original Filename: {filename}\n\n"
            output_text += "Decrypted Classification Results:\n"
            
            for label, certainty in results.items():
                output_text += f"**{label}**: {certainty:.2f}%\n"

            self._update_results_text(output_text)
            messagebox.showinfo("Decryption Success", f"Successfully decrypted results for {filename}.")
            
            # Optional: Display the original image (decoded from base64)
            image_data_b64 = decrypted_payload.get('image_data')
            if image_data_b64:
                image_bytes = base64.b64decode(image_data_b64)
                img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # Resize image for display purposes
                display_img = img.resize((300, 300), Image.LANCZOS)
                self.current_image_tk = ImageTk.PhotoImage(display_img)
                self.image_label.config(image=self.current_image_tk)
                self.image_label.image = self.current_image_tk 

        except ValueError as e:
            # Specifically catch the InvalidTag error from decryption
            messagebox.showerror("Decryption Error", f"Decryption failed: {e}")
            self._update_results_text(f"Decryption ERROR: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred during file processing: {e}")
            self._update_results_text(f"File Processing ERROR: {e}")

    # --- Helper Method ---
    def _update_results_text(self, text):
        """Helper to safely update the Text widget with results."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("640x480")
    app = DensenetClassifierApp(root)
    root.mainloop()