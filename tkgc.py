import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import os
import sys

class KeygenGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Keygen Generator")
        self.root.geometry("1200x800")
        
        # Create main frames
        self.create_input_frame()
        self.create_output_frame()
        self.create_button_frame()
        
        # Initialize model
        self.setup_model()

    def create_input_frame(self):
        # Input frame for protection code
        input_frame = ttk.LabelFrame(self.root, text="Protection Code")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # File loading controls
        file_frame = ttk.Frame(input_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Load File", 
                  command=self.load_file).pack(side=tk.LEFT, padx=5)
        
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, 
                 width=80).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Code input area
        self.code_text = scrolledtext.ScrolledText(input_frame, height=15)
        self.code_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_output_frame(self):
        # Output frame for generated keygen code
        output_frame = ttk.LabelFrame(self.root, text="Generated Keygen Code")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_button_frame(self):
        # Button controls
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Generate Keygen", 
                  command=self.generate_keygen).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save Keygen", 
                  command=self.save_keygen).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Test Keygen", 
                  command=self.test_keygen).pack(side=tk.LEFT, padx=5)

    def setup_model(self):
        try:
            self.output_text.insert(tk.END, "Loading DeepSeek model...\n")
            self.root.update()
            
            # Check GPU availability
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.output_text.insert(tk.END, f"Using device: {self.device}\n")
            
            model_path = "deepseek-ai/deepseek-coder-1.3b-base"
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir="./models",
                local_files_only=False
            )
            
            # Initialize model with proper device settings
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir="./models"
            )
            
            # Move model to GPU if available
            if self.device == 'cuda':
                self.model = self.model.cuda()
                self.output_text.insert(tk.END, "Model loaded successfully on GPU!\n")
            else:
                self.output_text.insert(tk.END, "Model loaded on CPU (GPU not available)\n")
            
        except Exception as e:
            self.output_text.insert(tk.END, f"Error loading model: {str(e)}\n")
            messagebox.showerror("Error", f"Failed to load DeepSeek model: {str(e)}")
    
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Protection Code File",
            filetypes=[
                ("All Files", "*.*"),
                ("C Files", "*.c"),
                ("C++ Files", "*.cpp"),
                ("Header Files", "*.h"),
                ("Assembly Files", "*.asm")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.code_text.delete('1.0', tk.END)
                self.code_text.insert('1.0', content)
                self.file_path_var.set(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def generate_keygen(self):
        protection_code = self.code_text.get('1.0', tk.END).strip()
        if not protection_code:
            messagebox.showwarning("Warning", "Please enter protection code first")
            return
            
        # Generate keygen code
        keygen_code = self.generate_keygen_code(protection_code)
        
        # Display generated code
        self.output_text.delete('1.0', tk.END)
        self.output_text.insert('1.0', keygen_code)
        
        messagebox.showinfo("Success", "Keygen code generated successfully!")

    def generate_keygen_code(self, protection_code):
        # Analyze protection scheme
        key_length = self.extract_key_length(protection_code)
        validation_checks = self.identify_validation_checks(protection_code)
        
        # Generate keygen GUI code
        return f"""import tkinter as tk
from tkinter import ttk, messagebox
import random
import struct

class ProtectionKeygen:
    def __init__(self, root):
        self.root = root
        self.root.title("Protection Keygen")
        self.root.geometry("500x400")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Protection Keygen", 
                              font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Key options frame
        options_frame = ttk.LabelFrame(main_frame, text="Key Options", padding="5")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Key length option
        length_frame = ttk.Frame(options_frame)
        length_frame.pack(fill=tk.X, pady=5)
        ttk.Label(length_frame, text="Key Length:").pack(side=tk.LEFT, padx=5)
        self.length_var = tk.IntVar(value={key_length})
        ttk.Entry(length_frame, textvariable=self.length_var, 
                 width=10).pack(side=tk.LEFT, padx=5)
        
        # Generated key frame
        key_frame = ttk.LabelFrame(main_frame, text="Generated Key", padding="5")
        key_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.key_var = tk.StringVar()
        key_entry = ttk.Entry(key_frame, textvariable=self.key_var, 
                            width=50, font=('Courier', 10))
        key_entry.pack(padx=5, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Generate Key", 
                  command=self.generate_key).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Copy to Clipboard",
                  command=self.copy_key).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Verify Key",
                  command=self.verify_key).pack(side=tk.LEFT, padx=5)
    
    def generate_key(self):
        try:
            length = self.length_var.get()
            if length < 1:
                raise ValueError("Invalid key length")
                
            # Generate key bytes
            key_bytes = []
            for _ in range(length):
                byte = random.randint(0, 255)
                key_bytes.append(byte)
            
            # Apply validation
            {self.generate_validation_code(validation_checks)}
            
            # Format key
            hex_str = ''.join(f'{{b:02X}}' for b in key_bytes)
            formatted = '-'.join(hex_str[i:i+4] for i in range(0, len(hex_str), 4))
            
            self.key_var.set(formatted)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate key: {{str(e)}}")
    
    def copy_key(self):
        key = self.key_var.get()
        if key:
            self.root.clipboard_clear()
            self.root.clipboard_append(key)
            messagebox.showinfo("Success", "Key copied to clipboard!")
        else:
            messagebox.showwarning("Warning", "Generate a key first!")
    
    def verify_key(self):
        key = self.key_var.get()
        if not key:
            messagebox.showwarning("Warning", "Generate a key first!")
            return
            
        try:
            # Verify key format and validation
            key_bytes = bytes.fromhex(key.replace('-', ''))
            if self.verify_validation(key_bytes):
                messagebox.showinfo("Success", "Key verification passed!")
            else:
                messagebox.showerror("Error", "Invalid key!")
        except Exception as e:
            messagebox.showerror("Error", f"Verification failed: {{str(e)}}")
    
    def verify_validation(self, key_bytes):
        # Implement validation checks
        try:
            {self.generate_verification_code(validation_checks)}
            return True
        except:
            return False

def main():
    root = tk.Tk()
    app = ProtectionKeygen(root)
    root.mainloop()

if __name__ == "__main__":
    main()
"""

    def save_keygen(self):
        if not self.output_text.get('1.0', tk.END).strip():
            messagebox.showwarning("Warning", "Generate keygen code first!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python Files", "*.py")],
            title="Save Keygen As"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.output_text.get('1.0', tk.END))
                messagebox.showinfo("Success", f"Keygen saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save keygen: {str(e)}")

    def test_keygen(self):
        if not self.output_text.get('1.0', tk.END).strip():
            messagebox.showwarning("Warning", "Generate keygen code first!")
            return
            
        try:
            # Save temporary file
            temp_file = "_temp_keygen.py"
            with open(temp_file, 'w') as f:
                f.write(self.output_text.get('1.0', tk.END))
            
            # Run keygen in a new process
            import subprocess
            subprocess.Popen([sys.executable, temp_file])
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to test keygen: {str(e)}")
        
        finally:
            # Clean up temp file after a short delay to ensure it's loaded
            self.root.after(1000, lambda: self.cleanup_temp_file(temp_file))

    def cleanup_temp_file(self, temp_file):
        """Clean up temporary keygen file"""
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            print(f"Failed to cleanup temp file: {str(e)}")

    def extract_key_length(self, code):
        """Extract required key length from code"""
        # Look for key_length parameter or buffer size
        length_patterns = [
            r'key_length.*?(\d+)',
            r'buffer\[(\d+)\]',
            r'length.*?(\d+)'
        ]
        
        for pattern in length_patterns:
            match = re.search(pattern, code)
            if match:
                return int(match.group(1))
        
        return 16  # Default length if not found

    def identify_validation_checks(self, code):
        """Identify validation checks from code"""
        checks = []
        
        if 'NULL' in code:
            checks.append("NULL pointer validation")
        
        if '0xffffffff' in code:
            checks.append("Error code validation")
        
        if 'FUN_180003300' in code:
            checks.append("Custom validation routine")
        
        if 'checksum' in code.lower():
            checks.append("Checksum validation")
        
        if len(checks) == 0:
            checks.append("No explicit validation found")
        
        return checks

    def generate_validation_code(self, checks):
        """Generate validation code based on identified checks"""
        code = []
        
        if "Checksum validation" in checks:
            code.append("""
            # Calculate and set checksum
            checksum = sum(key_bytes[:-1]) & 0xFF
            key_bytes[-1] = (256 - checksum) & 0xFF""")
        
        if "Custom validation routine" in checks:
            code.append("""
            # Custom validation
            for i in range(len(key_bytes)-1):
                key_bytes[i] ^= key_bytes[i+1]""")
        
        return '\n'.join(code) if code else "pass"

    def generate_verification_code(self, checks):
        # Implement verification code generation logic
        pass

def main():
    root = tk.Tk()
    app = KeygenGeneratorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
