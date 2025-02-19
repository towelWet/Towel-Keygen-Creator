import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import os
import sys
import time
import json
import google.generativeai as genai
import random
import struct
import hashlib

class KeygenGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Keygen Generator")
        self.root.geometry("1200x800")
        
        # Model settings
        self.model_var = tk.StringVar(value="deepseek")
        self.gemini_key = tk.StringVar()
        self.load_config()
        
        # Create menu
        self.create_menu()
        
        # Create main frames
        self.create_model_frame()
        self.create_input_frame()
        self.create_output_frame()
        self.create_button_frame()
        
        # Initialize selected model
        self.setup_selected_model()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Configure Gemini API", 
                                command=self.show_gemini_config)

    def create_model_frame(self):
        model_frame = ttk.LabelFrame(self.root, text="Model Selection")
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(model_frame, text="DeepSeek (Offline)", 
                       variable=self.model_var, value="deepseek",
                       command=self.setup_selected_model).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="Gemini API", 
                       variable=self.model_var, value="gemini",
                       command=self.setup_selected_model).pack(side=tk.LEFT, padx=5)

    def show_gemini_config(self):
        config_window = tk.Toplevel(self.root)
        config_window.title("Gemini API Configuration")
        config_window.geometry("400x150")
        
        ttk.Label(config_window, text="Gemini API Key:").pack(padx=5, pady=5)
        
        key_entry = ttk.Entry(config_window, textvariable=self.gemini_key, width=50)
        key_entry.pack(padx=5, pady=5)
        
        def save_config():
            self.save_config()
            config_window.destroy()
            self.setup_selected_model()
            
        ttk.Button(config_window, text="Save", 
                  command=save_config).pack(pady=10)
        
        # Center window
        config_window.transient(self.root)
        config_window.grab_set()
        self.root.wait_window(config_window)

    def load_config(self):
        """Load configuration including model preference"""
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r') as f:
                    config = json.load(f)
                    self.gemini_key.set(config.get('gemini_key', ''))
                    # Load and set last used model
                    last_model = config.get('last_model', 'deepseek')
                    self.model_var.set(last_model)
            else:
                # Default to DeepSeek if no config exists
                self.model_var.set('deepseek')
        except Exception as e:
            print(f"Error loading config: {e}")
            self.model_var.set('deepseek')

    def save_config(self):
        """Save configuration including model preference"""
        try:
            config = {
                'gemini_key': self.gemini_key.get(),
                'last_model': self.model_var.get()  # Save current model selection
            }
            with open('config.json', 'w') as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Error saving config: {e}")

    def setup_selected_model(self):
        """Setup selected model and save preference"""
        if self.model_var.get() == "deepseek":
            self.setup_deepseek()
        else:
            self.setup_gemini()
        # Save the selection whenever it changes
        self.save_config()

    def setup_deepseek(self):
        """Initialize DeepSeek model"""
        try:
            model_path = "deepseek-ai/deepseek-coder-1.3b-base"
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir="./models",
                local_files_only=True,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                cache_dir="./models",
                local_files_only=True
            )
            
            print(f"DeepSeek model loaded on {self.device}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load DeepSeek model: {str(e)}")

    def setup_gemini(self):
        """Initialize Gemini API"""
        try:
            if not self.gemini_key.get():
                messagebox.showwarning("Warning", "Please configure Gemini API key first")
                self.show_gemini_config()
                return
                
            # Configure the API
            genai.configure(api_key=self.gemini_key.get())
            
            # Initialize the model
            self.gemini_model = genai.GenerativeModel('gemini-pro')
            
            # Test the configuration with a simple prompt
            try:
                response = self.gemini_model.generate_content("Test connection")
                print("Gemini API configured successfully")
            except Exception as e:
                raise Exception(f"API key validation failed: {str(e)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to configure Gemini API: {str(e)}")
            # Fallback to DeepSeek
            self.model_var.set("deepseek")
            self.setup_deepseek()

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
        """Create output frame with retry functionality"""
        output_frame = ttk.LabelFrame(self.root, text="Generated Keygen Code")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add feedback frame
        feedback_frame = ttk.Frame(output_frame)
        feedback_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(feedback_frame, text="Issues with generated keygen:").pack(side=tk.LEFT, padx=5)
        
        self.feedback_var = tk.StringVar()
        feedback_entry = ttk.Entry(feedback_frame, textvariable=self.feedback_var, width=50)
        feedback_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        retry_btn = ttk.Button(feedback_frame, text="Try Again", command=self.retry_generation)
        retry_btn.pack(side=tk.LEFT, padx=5)
        
        # Output text area
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

    def analyze_protection(self, code):
        """Analyze code to detect protection scheme"""
        prompt = """Analyze this code and identify any protection scheme, key validation, or licensing checks.
If found, describe:
1. Key format and length
2. Validation algorithm
3. Any constants or patterns used
4. Success/failure conditions

Code to analyze:
{code}

Respond in this format:
PROTECTION_FOUND: yes/no
KEY_LENGTH: length in bytes or 'unknown'
VALIDATION_TYPE: type of validation or 'unknown'
ALGORITHM: brief description
CONSTANTS: any important constants
SUCCESS_CHECK: condition for success
"""
        
        try:
            inputs = self.tokenizer(prompt.format(code=code), return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    num_beams=5,
                    temperature=0.3
                )
            
            analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._parse_analysis(analysis)
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            return None

    def _parse_analysis(self, analysis):
        """Parse the analysis response"""
        results = {}
        for line in analysis.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                results[key.strip()] = value.strip()
        return results

    def generate_keygen(self):
        """Generate keygen using selected model"""
        code = self.code_text.get('1.0', tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "Please enter code to analyze")
            return
            
        self.output_text.delete('1.0', tk.END)
        self.output_text.insert('1.0', "Analyzing protection scheme...\n")
        self.root.update()
        
        try:
            if self.model_var.get() == "deepseek":
                keygen_code = self.generate_with_deepseek(code)
            else:
                keygen_code = self.generate_with_gemini(code)
                
            self.output_text.delete('1.0', tk.END)
            self.output_text.insert('1.0', keygen_code)
            
        except Exception as e:
            self.output_text.delete('1.0', tk.END)
            self.output_text.insert('1.0', f"Error generating keygen: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate keygen: {str(e)}")

    def generate_with_deepseek(self, code, feedback=None):
        """Generate keygen using DeepSeek with optional feedback"""
        prompt = f"""Create a Python tkinter GUI keygen for this protection code:

{code}

Requirements:
1. Create a working tkinter GUI keygen
2. Implement the exact key generation algorithm
3. Include all validation checks
4. Add copy-to-clipboard functionality
5. Show validation status
6. Add detailed comments explaining the logic
"""

        if feedback:
            prompt += f"\nPrevious issues to fix:\n{feedback}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2000,
                num_beams=5,
                temperature=0.7,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_with_gemini(self, code, feedback=None):
        """Generate keygen using Gemini API with optional feedback"""
        prompt = f"""Create a Python tkinter GUI keygen for this protection code:

{code}

Requirements:
1. Create a working tkinter GUI keygen
2. Implement the exact key generation algorithm
3. Include all validation checks
4. Add copy-to-clipboard functionality
5. Show validation status
6. Add detailed comments explaining the logic
"""

        if feedback:
            prompt += f"\nPrevious issues to fix:\n{feedback}"
            prompt += "\nPlease generate an improved version addressing these issues."

        response = self.gemini_model.generate_content(prompt)
        generated_text = response.text
        
        # Extract code from response
        if "```python" in generated_text:
            generated_text = generated_text.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_text:
            generated_text = generated_text.split("```")[1].strip()
        
        return generated_text

    def analyze_and_generate_protection_code(self, code):
        """Generate the protection-specific code"""
        base_code = """    def generate_key(self):
            try:
                # Get key length from input or use default
                try:
                    length = int(self.length_var.get())
                except:
                    length = 32  # Default length
                
                # Generate random bytes
                key_bytes = []
                for i in range(length):
                    byte = random.randint(0, 255)
                    key_bytes.append(byte)
                
                # Format key in readable hex format with dashes
                hex_str = ''.join(f'{b:02X}' for b in key_bytes)
                formatted_key = '-'.join(hex_str[i:i+4] for i in range(0, len(hex_str), 4))
                
                # Update key display
                self.key_var.set(formatted_key)
                self.status_var.set("Key generated successfully")
                
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                messagebox.showerror("Error", f"Failed to generate key: {str(e)}")

        def verify_key(self):
            key = self.key_var.get()
            if not key:
                messagebox.showwarning("Warning", "Generate a key first!")
                return
            
            try:
                # Remove dashes and convert to bytes
                key_bytes = bytes.fromhex(key.replace('-', ''))
                
                # Verify length
                if len(key_bytes) != int(self.length_var.get()):
                    raise ValueError("Invalid key length")
                    
                # Add any additional validation checks here
                
                messagebox.showinfo("Success", "Key verification passed!")
                self.status_var.set("Key verified successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Invalid key: {str(e)}")
                self.status_var.set("Key verification failed")

        def copy_key(self):
            key = self.key_var.get()
            if key:
                self.root.clipboard_clear()
                self.root.clipboard_append(key)
                messagebox.showinfo("Success", "Key copied to clipboard!")
            else:
                messagebox.showwarning("Warning", "Generate a key first!")
        """
        return base_code

    def generate_keygen_code(self, protection_code):
        """Generate complete keygen code"""
        return f"""import tkinter as tk
from tkinter import ttk, messagebox
import random
import struct
import hashlib
import sys
import os

class KeygenGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Key Generator")
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Key length input
        length_frame = ttk.Frame(main_frame)
        length_frame.pack(fill=tk.X, pady=5)
        ttk.Label(length_frame, text="Key Length (bytes):").pack(side=tk.LEFT, padx=5)
        self.length_var = tk.StringVar(value="32")
        ttk.Entry(length_frame, textvariable=self.length_var, width=10).pack(side=tk.LEFT)
        
        # Key display
        key_frame = ttk.LabelFrame(main_frame, text="Generated Key", padding="5")
        key_frame.pack(fill=tk.X, pady=5)
        
        self.key_var = tk.StringVar()
        key_entry = ttk.Entry(key_frame, textvariable=self.key_var, width=50)
        key_entry.pack(padx=5, pady=5)
        
        # Status display
        self.status_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.status_var).pack(pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Generate Key", 
                  command=self.generate_key).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Copy Key",
                  command=self.copy_key).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Verify Key",
                  command=self.verify_key).pack(side=tk.LEFT, padx=5)

{self.analyze_and_generate_protection_code(protection_code)}

def main():
    root = tk.Tk()
    app = KeygenGUI(root)
    root.mainloop()

if __name__ == '__main__':
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

    def retry_generation(self):
        """Retry keygen generation with feedback"""
        code = self.code_text.get('1.0', tk.END).strip()
        feedback = self.feedback_var.get().strip()
        
        if not code:
            messagebox.showwarning("Warning", "Please enter code to analyze")
            return
        
        self.output_text.delete('1.0', tk.END)
        self.output_text.insert('1.0', "Regenerating keygen with feedback...\n")
        self.root.update()
        
        try:
            if self.model_var.get() == "deepseek":
                keygen_code = self.generate_with_deepseek(code, feedback)
            else:
                keygen_code = self.generate_with_gemini(code, feedback)
            
            self.output_text.delete('1.0', tk.END)
            self.output_text.insert('1.0', keygen_code)
            messagebox.showinfo("Success", "Keygen regenerated with feedback!")
            
        except Exception as e:
            self.output_text.delete('1.0', tk.END)
            self.output_text.insert('1.0', f"Error regenerating keygen: {str(e)}")
            messagebox.showerror("Error", f"Failed to regenerate keygen: {str(e)}")

def main():
    root = tk.Tk()
    app = KeygenGeneratorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
