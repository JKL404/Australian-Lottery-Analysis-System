#!/usr/bin/env python3
"""
Simple GUI for the Lottery Analysis System
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import subprocess
import threading
import os
import sys
from pathlib import Path
from datetime import datetime

class LotteryAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Australian Lottery Analyzer")
        root.geometry("800x700")  # Increased size to accommodate new elements
        root.resizable(True, True)
        
        # Set up the UI
        self.setup_ui()
    
    def setup_ui(self):
        # Header
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.pack(fill="x")
        
        ttk.Label(
            header_frame, 
            text="Australian Lottery Analyzer", 
            font=("Arial", 18, "bold")
        ).pack()
        
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10)
        
        # Main content
        content_frame = ttk.Frame(self.root, padding="20")
        content_frame.pack(fill="both", expand=True)
        
        # Create left and right frames for better organization
        left_frame = ttk.Frame(content_frame)
        left_frame.grid(row=0, column=0, sticky="nw", padx=(0, 20))
        
        right_frame = ttk.Frame(content_frame)
        right_frame.grid(row=0, column=1, sticky="nw")
        
        # Lottery selection (left frame)
        ttk.Label(
            left_frame, 
            text="Select Lottery Type:", 
            font=("Arial", 12)
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        self.lottery_var = tk.StringVar()
        lottery_types = [
            ("Powerball", "powerball"),
            ("Saturday Lotto", "saturday_lotto"),
            ("Oz Lotto", "oz_lotto"),
            ("Set For Life", "set_for_life"),
            ("Monday Lotto", "monday_lotto"),
            ("Wednesday Lotto", "wednesday_lotto")
        ]
        
        self.lottery_var.set(lottery_types[0][1])  # Default to first option
        
        for i, (label, value) in enumerate(lottery_types):
            ttk.Radiobutton(
                left_frame, 
                text=label, 
                value=value, 
                variable=self.lottery_var
            ).grid(row=i+1, column=0, sticky="w", padx=20, pady=5)
        
        # Options (right frame)
        ttk.Label(
            right_frame, 
            text="Analysis Options:", 
            font=("Arial", 12)
        ).grid(row=0, column=0, sticky="w", pady=(0, 10), columnspan=2)
        
        # Years selection
        ttk.Label(
            right_frame, 
            text="Years of Data:", 
        ).grid(row=1, column=0, sticky="w", pady=5)
        
        self.years_var = tk.IntVar()
        self.years_var.set(2)  # Default to 2 years
        
        years_spin = ttk.Spinbox(
            right_frame, 
            from_=1, 
            to=10, 
            textvariable=self.years_var, 
            width=5
        )
        years_spin.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Prediction count selection
        ttk.Label(
            right_frame, 
            text="Number of Predictions:", 
        ).grid(row=2, column=0, sticky="w", pady=5)
        
        self.predictions_var = tk.IntVar()
        self.predictions_var.set(10)  # Default to 10 predictions
        
        predictions_spin = ttk.Spinbox(
            right_frame, 
            from_=1, 
            to=50, 
            textvariable=self.predictions_var, 
            width=5
        )
        predictions_spin.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Prediction model
        ttk.Label(
            right_frame, 
            text="Prediction Model:", 
        ).grid(row=3, column=0, sticky="w", pady=5)
        
        self.model_var = tk.StringVar()
        self.model_var.set("advanced")  # Default to advanced
        
        model_combo = ttk.Combobox(
            right_frame,
            textvariable=self.model_var,
            values=["frequency", "time_weighted", "pattern", "advanced"],
            width=15,
            state="readonly"
        )
        model_combo.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        # Output format
        ttk.Label(
            right_frame, 
            text="Output Format:", 
        ).grid(row=4, column=0, sticky="w", pady=5)
        
        self.format_var = tk.StringVar()
        self.format_var.set("csv")  # Default to CSV
        
        format_frame = ttk.Frame(right_frame)
        format_frame.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Radiobutton(
            format_frame, 
            text="CSV", 
            value="csv", 
            variable=self.format_var
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            format_frame, 
            text="JSON", 
            value="json", 
            variable=self.format_var
        ).pack(side=tk.LEFT)
        
        # Visualization options
        ttk.Label(
            right_frame, 
            text="Visualization:", 
        ).grid(row=5, column=0, sticky="w", pady=5)
        
        self.visualize_var = tk.BooleanVar()
        self.visualize_var.set(True)  # Default to enabled
        
        self.interactive_var = tk.BooleanVar()
        self.interactive_var.set(False)  # Default to disabled
        
        viz_frame = ttk.Frame(right_frame)
        viz_frame.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Checkbutton(
            viz_frame, 
            text="Enable", 
            variable=self.visualize_var,
            command=self.toggle_interactive
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.interactive_check = ttk.Checkbutton(
            viz_frame, 
            text="Interactive", 
            variable=self.interactive_var
        )
        self.interactive_check.pack(side=tk.LEFT)
        
        # Generate report option
        ttk.Label(
            right_frame, 
            text="Generate Detailed Report:", 
        ).grid(row=6, column=0, sticky="w", pady=5)
        
        self.report_var = tk.BooleanVar()
        self.report_var.set(True)  # Default to enabled
        
        ttk.Checkbutton(
            right_frame, 
            text="Enable", 
            variable=self.report_var
        ).grid(row=6, column=1, sticky="w", padx=5, pady=5)
        
        # Actions section (spans both left and right)
        action_frame = ttk.LabelFrame(content_frame, text="Actions", padding="10")
        action_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=20)
        
        # Create a button frame for horizontal layout
        button_frame = ttk.Frame(action_frame)
        button_frame.pack(fill="x")
        
        self.analyze_button = ttk.Button(
            button_frame, 
            text="Run Analysis & Prediction", 
            command=self.run_analysis
        )
        self.analyze_button.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5), pady=5)
        
        self.view_predictions_button = ttk.Button(
            button_frame, 
            text="View Latest Predictions", 
            command=self.view_predictions
        )
        self.view_predictions_button.pack(side=tk.LEFT, fill="x", expand=True, padx=5, pady=5)
        
        self.view_analysis_button = ttk.Button(
            button_frame, 
            text="View Analysis Reports", 
            command=self.view_analysis
        )
        self.view_analysis_button.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 0), pady=5)
        
        # Log output
        log_frame = ttk.LabelFrame(content_frame, text="Progress Log", padding="10")
        log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=10)
        content_frame.grid_rowconfigure(2, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=80, wrap=tk.WORD)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.config(state="disabled")
        
        # Progress and status
        self.progress = ttk.Progressbar(
            self.root, 
            orient="horizontal", 
            mode="indeterminate"
        )
        self.progress.pack(fill="x", padx=20, pady=10)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_label = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            font=("Arial", 10, "italic")
        )
        self.status_label.pack(pady=(0, 20))
    
    def toggle_interactive(self):
        """Enable/disable interactive checkbox based on visualize state"""
        if self.visualize_var.get():
            self.interactive_check.config(state="normal")
        else:
            self.interactive_var.set(False)
            self.interactive_check.config(state="disabled")
    
    def log_message(self, message):
        """Add a message to the log window"""
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)  # Scroll to the end
        self.log_text.config(state="disabled")
    
    def run_analysis(self):
        """Run the analysis and prediction process"""
        lottery = self.lottery_var.get()
        years = self.years_var.get()
        
        # Clear log
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")
        
        # Disable buttons during processing
        self.analyze_button.config(state="disabled")
        self.view_predictions_button.config(state="disabled")
        self.view_analysis_button.config(state="disabled")
        
        # Start progress bar
        self.progress.start()
        self.status_var.set(f"Running analysis for {lottery.replace('_', ' ').title()}...")
        
        # Log the start of the process
        self.log_message(f"Starting analysis for {lottery.replace('_', ' ').title()} with {years} years of data")
        self.log_message(f"Generating {self.predictions_var.get()} predictions using the {self.model_var.get()} model")
        self.log_message("Process started at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.log_message("-" * 50)
        
        # Run the analysis in a separate thread
        thread = threading.Thread(
            target=self._run_analysis_thread, 
            args=(lottery, years)
        )
        thread.daemon = True
        thread.start()
    
    def _run_analysis_thread(self, lottery, years):
        """Thread function to run the analysis"""
        try:
            # Prepare the command
            cmd = [
                sys.executable, 
                "lotto_cli.py", 
                "--lottery", lottery, 
                "--action", "batch", 
                "--years", str(years),
                "--prediction-count", str(self.predictions_var.get()),
                "--model", self.model_var.get(),
                "--format", self.format_var.get()
            ]
            
            # Add optional flags
            if self.visualize_var.get():
                cmd.append("--visualize")
                
                if self.interactive_var.get():
                    cmd.append("--interactive")
            
            if self.report_var.get():
                cmd.append("--output-report")
            
            # Log the command
            self.log_message(f"Executing: {' '.join(cmd)}")
            self.log_message("-" * 50)
            
            # Execute the command - use universal_newlines for text mode
            # and line buffering to get immediate output
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout to capture all output
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            # Process output in real-time
            for line in process.stdout:
                # Strip and check if line is not empty
                line_text = line.strip()
                if line_text:
                    # Update status with the current task for certain key phrases
                    if any(keyword in line_text for keyword in ["STEP", "Starting", "Generating", "Analyzing", "Scraping", "Processing", "Saving"]):
                        self.status_var.set(line_text)
                    
                    # Log all output
                    self.log_message(line_text)
                
                # Update the GUI regularly to prevent freezing
                self.root.update_idletasks()
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Check for errors
            if return_code != 0:
                self.log_message(f"Process exited with error code {return_code}")
                raise Exception(f"Process exited with error code {return_code}")
            
            # Process completed successfully
            self.status_var.set(f"Analysis complete for {lottery.replace('_', ' ').title()}")
            self.log_message("-" * 50)
            self.log_message(f"Analysis completed successfully!")
            self.log_message(f"Process finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            messagebox.showinfo(
                "Analysis Complete", 
                f"The analysis and prediction for {lottery.replace('_', ' ').title()} is complete.\n\n"
                f"Click 'View Latest Predictions' to see the prediction results or "
                f"'View Analysis Reports' to see the detailed analysis."
            )
            
        except Exception as e:
            error_msg = str(e)
            self.status_var.set(f"Error: {error_msg}")
            self.log_message(f"ERROR: {error_msg}")
            messagebox.showerror("Error", f"An error occurred: {error_msg}")
        
        finally:
            # Stop progress bar and re-enable buttons
            self.progress.stop()
            self.analyze_button.config(state="normal")
            self.view_predictions_button.config(state="normal")
            self.view_analysis_button.config(state="normal")
    
    def view_predictions(self):
        """Open the predictions folder"""
        predictions_path = Path("results/predictions")
        self._open_folder(predictions_path)
    
    def view_analysis(self):
        """Open the analysis folder"""
        analysis_path = Path("results/analysis")
        self._open_folder(analysis_path)
    
    def _open_folder(self, path):
        """Open a folder in the system file explorer"""
        if not path.exists():
            messagebox.showwarning(
                "Folder Not Found", 
                f"The folder {path} does not exist yet. Run an analysis first."
            )
            return
        
        try:
            import platform
            system = platform.system()
            
            if system == "Windows":
                os.startfile(str(path))
            elif system == "Darwin":  # macOS
                subprocess.call(["open", str(path)])
            else:  # Linux
                subprocess.call(["xdg-open", str(path)])
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {str(e)}")

def main():
    root = tk.Tk()
    app = LotteryAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 