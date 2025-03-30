#!/usr/bin/env python3
"""
Enhanced GUI for the Australian Lottery Analysis System
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import subprocess
import threading
import os
import sys
import time
import platform
from pathlib import Path
from datetime import datetime
import socket
import subprocess
import json
import webbrowser
import shutil
import tempfile
import re
import csv
import io

class LotteryAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Australian Lottery Analyzer")
        root.geometry("800x700")
        root.resizable(True, True)
        
        # Add app icon if available
        self.set_app_icon()
        
        # Theme settings
        self.theme_var = tk.StringVar(value="light")  # Default to light theme
        
        # Configure styles - apply initial theme before loading settings
        self.configure_styles()
        
        # Track processing state
        self.is_processing = False
        self.start_time = None
        
        # Set up the UI first
        self.setup_ui()
        
        # Load and apply saved settings AFTER UI is set up
        self.load_settings()
        
        # Set up keyboard shortcuts
        self.setup_shortcuts()
        
        # Automatically check for results directories
        self.check_results_directories()
    
    def set_app_icon(self):
        """Set the application icon if available"""
        icon_path = Path("resources/icon.png")
        if icon_path.exists():
            try:
                # Platform-specific icon setting
                if platform.system() == "Windows":
                    self.root.iconbitmap(str(icon_path.with_suffix('.ico')))
                else:
                    logo = tk.PhotoImage(file=str(icon_path))
                    self.root.iconphoto(True, logo)
            except Exception:
                pass  # Silently fail if icon can't be set
    
    def configure_styles(self):
        """Configure custom styles for the application"""
        style = ttk.Style()
        
        # Set initial theme based on preference
        self.apply_theme(self.theme_var.get())
        
        # Configure button styles with more modern look
        style.configure("TButton", 
                       font=("Arial", 10),
                       padding=(10, 6))
        
        # Create a prominent style for the analysis button
        style.configure("Analysis.TButton", 
                       font=("Arial", 11, "bold"),
                       padding=(15, 8))
        
        # Try to set background color if supported by the theme
        try:
            style.configure("Analysis.TButton", background="#4CAF50", foreground="#ffffff")
        except:
            pass  # Some themes don't support background colors for ttk buttons
        
        # Create accent button style
        style.configure("Accent.TButton",
                       font=("Arial", 10),
                       padding=(10, 6))
        try:
            style.configure("Accent.TButton", background="#3498db", foreground="#ffffff")
        except:
            pass
        
        # Configure label styles with improved typography
        style.configure("Header.TLabel", font=("Arial", 20, "bold"))
        style.configure("Section.TLabel", font=("Arial", 12, "bold"))
        style.configure("Status.TLabel", font=("Arial", 10, "italic"))
        
        # Configure frame styles for better visual separation
        style.configure("Card.TFrame", padding=15)
        
        # Configure notebook tab style
        style.configure("TNotebook.Tab", padding=(12, 6), font=("Arial", 10))
    
    def apply_theme(self, theme_name):
        """Apply the selected theme with enhanced color schemes"""
        style = ttk.Style()
        
        if theme_name == "dark":
            # Set ttk theme to a darker option if available
            try:
                style.theme_use("clam")  # clam is darker than default
            except:
                pass
                
            # Configure dark colors for the log text area with improved contrast
            if hasattr(self, 'log_text'):
                self.log_text.config(bg="#1E1E1E", fg="#E0E0E0")
                self.log_text.tag_configure("info", foreground="#E0E0E0")        # Light gray
                self.log_text.tag_configure("success", foreground="#8BC34A")     # Light green
                self.log_text.tag_configure("warning", foreground="#FFC107")     # Brighter yellow
                self.log_text.tag_configure("error", foreground="#FF5252")       # Brighter red
                self.log_text.tag_configure("command", foreground="#64B5F6")     # Light blue
                self.log_text.tag_configure("header", foreground="#CE93D8")      # Light purple
            
            # Root window background (if supported)
            try:
                self.root.configure(background="#2D2D2D")
            except:
                pass
                
            # Dark theme for main elements
            try:
                style.configure("TFrame", background="#2D2D2D")
                style.configure("TLabel", background="#2D2D2D", foreground="#E0E0E0")
                style.configure("TNotebook", background="#1E1E1E", borderwidth=0)
                style.configure("TNotebook.Tab", background="#3D3D3D", foreground="#E0E0E0")
                style.map("TNotebook.Tab", 
                        background=[("selected", "#1976D2")],
                        foreground=[("selected", "#FFFFFF")])
                style.configure("TButton", background="#3D3D3D", foreground="#E0E0E0")
                style.map("TButton", 
                        background=[("active", "#505050")],
                        foreground=[("active", "#FFFFFF")])
                style.configure("TCheckbutton", background="#2D2D2D", foreground="#E0E0E0")
                style.configure("TRadiobutton", background="#2D2D2D", foreground="#E0E0E0")
                style.configure("TLabelframe", background="#2D2D2D", foreground="#E0E0E0")
                style.configure("TLabelframe.Label", background="#2D2D2D", foreground="#E0E0E0")
            except:
                # Some styles might not be configurable depending on the ttk theme
                pass
                
        else:  # Light theme with a more modern look
            # Set ttk theme to a lighter option
            try:
                style.theme_use("default")
            except:
                pass
                
            # Configure light colors for the log text area with a soft paper look
            if hasattr(self, 'log_text'):
                self.log_text.config(bg="#F5F5F5", fg="#333333")
                self.log_text.tag_configure("info", foreground="#333333")       # Dark gray
                self.log_text.tag_configure("success", foreground="#2E7D32")    # Forest green
                self.log_text.tag_configure("warning", foreground="#F57C00")    # Brighter orange
                self.log_text.tag_configure("error", foreground="#C62828")      # Brighter red
                self.log_text.tag_configure("command", foreground="#1565C0")    # Navy blue
                self.log_text.tag_configure("header", foreground="#6A1B9A")     # Muted purple
            
            # Root window background (if supported)
            try:
                self.root.configure(background="#F5F5F5")
            except:
                pass
                
            # Light theme for main elements
            try:
                style.configure("TFrame", background="#F5F5F5")
                style.configure("TLabel", background="#F5F5F5", foreground="#333333")
                style.configure("TNotebook", background="#F5F5F5", borderwidth=0)
                style.configure("TNotebook.Tab", background="#E0E0E0", foreground="#333333")
                style.map("TNotebook.Tab", 
                        background=[("selected", "#2196F3")],
                        foreground=[("selected", "#FFFFFF")])
                style.configure("TButton", background="#E0E0E0", foreground="#333333")
                style.map("TButton", 
                        background=[("active", "#D0D0D0")],
                        foreground=[("active", "#000000")])
                style.configure("TCheckbutton", background="#F5F5F5", foreground="#333333")
                style.configure("TRadiobutton", background="#F5F5F5", foreground="#333333")
                style.configure("TLabelframe", background="#F5F5F5", foreground="#333333")
                style.configure("TLabelframe.Label", background="#F5F5F5", foreground="#333333")
            except:
                # Some styles might not be configurable depending on the ttk theme
                pass
    
    def setup_shortcuts(self):
        """Set up keyboard shortcuts"""
        self.root.bind("<F5>", lambda event: self.run_analysis())
        self.root.bind("<F6>", lambda event: self.view_predictions())
        self.root.bind("<F7>", lambda event: self.view_analysis())
        self.root.bind("<Control-s>", lambda event: self.save_logs())
        self.root.bind("<Escape>", lambda event: self.confirm_quit() if not self.is_processing else None)
    
    def setup_ui(self):
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=15, pady=15)  # Increased padding
        
        # Create main tab
        self.main_tab = ttk.Frame(self.notebook, style="Card.TFrame")
        self.notebook.add(self.main_tab, text="Lottery Analysis")
        
        # Create help tab
        self.help_tab = ttk.Frame(self.notebook, style="Card.TFrame")
        self.notebook.add(self.help_tab, text="Help")
        self.setup_help_tab()
        
        # Add settings tab
        self.settings_tab = ttk.Frame(self.notebook, style="Card.TFrame")
        self.notebook.add(self.settings_tab, text="Settings")
        self.setup_settings_tab()
        
        # Header with more prominent styling
        header_frame = ttk.Frame(self.main_tab, padding="20")  # Increased padding
        header_frame.pack(fill="x")
        
        ttk.Label(
            header_frame, 
            text="Australian Lottery Analyzer", 
            style="Header.TLabel"
        ).pack()
        
        ttk.Separator(self.main_tab, orient="horizontal").pack(fill="x", padx=20, pady=10)  # Added padding
        
        # Main content with improved spacing
        content_frame = ttk.Frame(self.main_tab, padding="25")  # Increased padding
        content_frame.pack(fill="both", expand=True)
        
        # Create left and right frames with more spacing
        left_frame = ttk.Frame(content_frame, style="Card.TFrame")
        left_frame.grid(row=0, column=0, sticky="nw", padx=(0, 25))  # Increased spacing
        
        right_frame = ttk.Frame(content_frame, style="Card.TFrame")
        right_frame.grid(row=0, column=1, sticky="nw")
        
        # Lottery selection (left frame)
        ttk.Label(
            left_frame, 
            text="Select Lottery Type:", 
            style="Section.TLabel"
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
            radio = ttk.Radiobutton(
                left_frame, 
                text=label, 
                value=value, 
                variable=self.lottery_var
            )
            radio.grid(row=i+1, column=0, sticky="w", padx=20, pady=5)
            self.create_tooltip(radio, f"Analyze {label} lottery data")
        
        # Options (right frame)
        ttk.Label(
            right_frame, 
            text="Analysis Options:", 
            style="Section.TLabel"
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
        self.create_tooltip(years_spin, "Number of years of historical data to analyze")
        
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
        self.create_tooltip(predictions_spin, "Number of lottery number combinations to generate")
        
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
        self.create_tooltip(model_combo, "Advanced: uses all methods combined\nFrequency: based on historical frequency\nTime weighted: emphasizes recent draws\nPattern: based on detected patterns")
        
        # Output format
        ttk.Label(
            right_frame, 
            text="Output Format:", 
        ).grid(row=4, column=0, sticky="w", pady=5)
        
        self.format_var = tk.StringVar()
        self.format_var.set("csv")  # Default to CSV
        
        format_frame = ttk.Frame(right_frame)
        format_frame.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
        csv_radio = ttk.Radiobutton(
            format_frame, 
            text="CSV", 
            value="csv", 
            variable=self.format_var
        )
        csv_radio.pack(side=tk.LEFT, padx=(0, 10))
        self.create_tooltip(csv_radio, "Save predictions as CSV (spreadsheet format)")
        
        json_radio = ttk.Radiobutton(
            format_frame, 
            text="JSON", 
            value="json", 
            variable=self.format_var
        )
        json_radio.pack(side=tk.LEFT)
        self.create_tooltip(json_radio, "Save predictions as JSON (structured data format)")
        
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
        
        viz_check = ttk.Checkbutton(
            viz_frame, 
            text="Enable", 
            variable=self.visualize_var,
            command=self.toggle_interactive
        )
        viz_check.pack(side=tk.LEFT, padx=(0, 10))
        self.create_tooltip(viz_check, "Generate charts and visualizations during analysis")
        
        self.interactive_check = ttk.Checkbutton(
            viz_frame, 
            text="Interactive", 
            variable=self.interactive_var
        )
        self.interactive_check.pack(side=tk.LEFT)
        self.create_tooltip(self.interactive_check, "Generate interactive web-based visualizations (requires plotly)")
        
        # Generate report option
        ttk.Label(
            right_frame, 
            text="Generate Detailed Report:", 
        ).grid(row=6, column=0, sticky="w", pady=5)
        
        self.report_var = tk.BooleanVar()
        self.report_var.set(True)  # Default to enabled
        
        report_check = ttk.Checkbutton(
            right_frame, 
            text="Enable", 
            variable=self.report_var
        )
        report_check.grid(row=6, column=1, sticky="w", padx=5, pady=5)
        self.create_tooltip(report_check, "Generate a comprehensive PDF report with analysis results")
        
        # Browse option output directory
        ttk.Label(
            right_frame, 
            text="Save Results To:", 
        ).grid(row=7, column=0, sticky="w", pady=5)
        
        self.output_dir_var = tk.StringVar()
        self.output_dir_var.set("results")  # Default directory
        
        dir_frame = ttk.Frame(right_frame)
        dir_frame.grid(row=7, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Entry(
            dir_frame,
            textvariable=self.output_dir_var,
            width=15
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        browse_btn = ttk.Button(
            dir_frame,
            text="Browse...",
            command=self.browse_output_dir
        )
        browse_btn.pack(side=tk.LEFT)
        self.create_tooltip(browse_btn, "Select where to save analysis results and predictions")
        
        # Actions section with improved visual design
        action_frame = ttk.LabelFrame(content_frame, text="Actions", padding="15")  # Increased padding
        action_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=25)  # Increased spacing
        
        # Button frame with better spacing
        button_frame = ttk.Frame(action_frame)
        button_frame.pack(fill="x", pady=5)  # Added padding
        
        # Try to use ttk button with styling first
        use_ttk_button = True
        try:
            style = ttk.Style()
            style.configure("Analysis.TButton", 
                           font=("Arial", 11, "bold"),
                           padding=(10, 5))
        except:
            use_ttk_button = False
        
        # Create either a ttk button or a regular tk button based on styling support
        if use_ttk_button:
            self.analyze_button = ttk.Button(
                button_frame, 
                text="▶ Run Analysis & Prediction (F5)", 
                command=self.run_analysis,
                style="Analysis.TButton"
            )
        else:
            # Fallback to regular tk button which has better styling support
            self.analyze_button = tk.Button(
                button_frame, 
                text="▶ Run Analysis & Prediction (F5)", 
                command=self.run_analysis,
                font=("Arial", 11, "bold"),
                bg="#4CAF50",  # Green background
                fg="white",    # White text
                padx=10,
                pady=5,
                relief=tk.RAISED,
                borderwidth=2
            )
        
        self.analyze_button.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5), pady=8)
        self.create_tooltip(self.analyze_button, "Start the analysis and prediction process")
        
        self.view_predictions_button = ttk.Button(
            button_frame, 
            text="View Latest Predictions (F6)", 
            command=self.view_predictions
        )
        self.view_predictions_button.pack(side=tk.LEFT, fill="x", expand=True, padx=5, pady=5)
        self.create_tooltip(self.view_predictions_button, "Open the folder containing prediction results")
        
        self.view_analysis_button = ttk.Button(
            button_frame, 
            text="View Analysis Reports (F7)", 
            command=self.view_analysis
        )
        self.view_analysis_button.pack(side=tk.LEFT, fill="x", expand=True, padx=(5, 0), pady=5)
        self.create_tooltip(self.view_analysis_button, "Open the folder containing analysis reports and visualizations")
        
        # Improved Log frame styling
        log_frame = ttk.LabelFrame(content_frame, text="Progress Log", padding="15")  # Increased padding
        log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=15)  # Increased spacing
        content_frame.grid_rowconfigure(2, weight=1)
        
        # Toolbar for log operations
        log_toolbar = ttk.Frame(log_frame)
        log_toolbar.pack(fill="x", pady=(0, 5))
        
        self.clear_log_button = ttk.Button(
            log_toolbar,
            text="Clear Log",
            command=self.clear_logs,
            width=10
        )
        self.clear_log_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.clear_log_button, "Clear the log window")
        
        self.save_log_button = ttk.Button(
            log_toolbar,
            text="Save Log",
            command=self.save_logs,
            width=10
        )
        self.save_log_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.save_log_button, "Save log contents to a file")
        
        # Show elapsed time
        self.elapsed_var = tk.StringVar()
        self.elapsed_var.set("Elapsed: 00:00:00")
        
        ttk.Label(
            log_toolbar,
            textvariable=self.elapsed_var,
            style="Status.TLabel"
        ).pack(side=tk.RIGHT, padx=5)
        
        # Log text area with custom tags for coloring - PAPER THEME
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=80, wrap=tk.WORD)
        self.log_text.pack(fill="both", expand=True)
        
        # Set background color to a soft cream color like old paper
        self.log_text.config(state="normal", bg="#F8F5E6", fg="#333333")  # Cream background, dark gray text
        
        # Configure text tags with softer colors that work well with cream background
        self.log_text.tag_configure("info", foreground="#333333")  # Dark gray for basic info
        self.log_text.tag_configure("success", foreground="#2E7D32")  # Forest green
        self.log_text.tag_configure("warning", foreground="#B45F04")  # Brown-orange
        self.log_text.tag_configure("error", foreground="#A52A2A")  # Brick red
        self.log_text.tag_configure("command", foreground="#1A478A")  # Navy blue
        self.log_text.tag_configure("header", foreground="#5D3954")  # Muted purple
        
        # Make the log text read-only
        self.log_text.config(state="disabled")
        
        # Progress and status
        self.progress = ttk.Progressbar(
            self.main_tab, 
            orient="horizontal", 
            mode="indeterminate"
        )
        self.progress.pack(fill="x", padx=20, pady=10)
        
        # Status bar with multiple elements
        status_frame = ttk.Frame(self.main_tab)
        status_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        self.status_label = ttk.Label(
            status_frame, 
            textvariable=self.status_var, 
            style="Status.TLabel"
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Current output directory indicator on right side
        self.output_status_var = tk.StringVar()
        self.update_output_status()
        
        ttk.Label(
            status_frame,
            textvariable=self.output_status_var,
            style="Status.TLabel"
        ).pack(side=tk.RIGHT)
        
        # Start timer for updating elapsed time
        self.update_timer()
    
    def setup_help_tab(self):
        """Set up the help tab content"""
        # Create scrollable frame for help content
        help_frame = ttk.Frame(self.help_tab, padding="20")
        help_frame.pack(fill="both", expand=True)
        
        help_title = ttk.Label(
            help_frame,
            text="Australian Lottery Analyzer - Help",
            style="Header.TLabel"
        )
        help_title.pack(pady=(0, 20))
        
        # Create text widget for help content
        help_text = scrolledtext.ScrolledText(
            help_frame, 
            wrap=tk.WORD, 
            width=70, 
            height=25,
            font=("Arial", 10)
        )
        help_text.pack(fill="both", expand=True)
        
        # Insert help content
        help_content = """
KEYBOARD SHORTCUTS
-----------------
F5: Run Analysis & Prediction
F6: View Latest Predictions
F7: View Analysis Reports
Ctrl+S: Save Logs
ESC: Quit Application (when not processing)

LOTTERY TYPES
------------
- Powerball: Draw with main numbers and a Powerball number
- Saturday Lotto: Traditional Saturday night draw
- Oz Lotto: Tuesday night draw with higher jackpots
- Set For Life: Daily draw with ongoing payments
- Monday Lotto: Monday night draw
- Wednesday Lotto: Wednesday night draw

ANALYSIS OPTIONS
--------------
Years of Data: Number of years of historical data to analyze (1-10)
Number of Predictions: How many prediction combinations to generate (1-50)

Prediction Models:
- Advanced: Combined approach using all methods (recommended)
- Frequency: Based on historical frequency of numbers
- Time Weighted: Places more emphasis on recent draws
- Pattern: Based on detected patterns in historical draws

Output Format:
- CSV: Standard spreadsheet format (Excel-compatible)
- JSON: Structured data format for advanced users

Visualization Options:
- Enable: Generate charts and graphs showing patterns
- Interactive: Create web-based interactive visualizations

Generate Detailed Report:
- Creates a comprehensive PDF report with analysis findings

BROWSER NOTE
-----------
During data scraping, Firefox will open automatically.
If you see a message that the lottery is not available in your state (especially in QLD),
quickly switch to NSW within the browser.

ANALYSIS PROCESS
--------------
The analysis process consists of several steps:
1. Importing any existing historical data
2. Scraping recent lottery results
3. Analyzing number patterns and frequencies
4. Generating predictions based on models
5. Creating visualizations and reports

Results are saved in the specified output directory in separate folders:
- predictions: Contains generated number combinations
- analysis: Contains reports and visualizations
- historical: Contains processed historical data

DISCLAIMER
---------
This software is for entertainment purposes only.
Lottery outcomes are random and no prediction system can guarantee winnings.
"""
        help_text.insert(tk.END, help_content)
        help_text.config(state="disabled")  # Make read-only
    
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget with improved styling"""
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            
            # Create tooltip window with enhanced styling
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            # Apply theme-aware styling
            if self.theme_var.get() == "dark":
                bg_color = "#424242"
                fg_color = "#E0E0E0"
            else:
                bg_color = "#FFFFF0"  # Soft ivory color
                fg_color = "#333333"
            
            self.tooltip.configure(background=bg_color)
            
            # Add subtle shadow effect (frame within frame)
            frame = tk.Frame(self.tooltip, background=bg_color, bd=1, relief=tk.SOLID)
            frame.pack(fill="both", expand=True)
            
            label = ttk.Label(
                frame, 
                text=text, 
                background=bg_color, 
                foreground=fg_color,
                font=("Arial", 9),
                justify=tk.LEFT,
                padding=8,
                wraplength=250
            )
            label.pack()
            
        def leave(event):
            if hasattr(self, "tooltip"):
                self.tooltip.destroy()
                
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
    
    def update_timer(self):
        """Update the elapsed time display"""
        if self.is_processing and self.start_time:
            elapsed = time.time() - self.start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.elapsed_var.set(f"Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Schedule the next update
        self.root.after(1000, self.update_timer)
    
    def update_output_status(self):
        """Update the output directory status display"""
        self.output_status_var.set(f"Output: {self.output_dir_var.get()}")
    
    def check_results_directories(self):
        """Check if results directories exist and update button states"""
        predictions_path = Path(self.output_dir_var.get()) / "predictions"
        analysis_path = Path(self.output_dir_var.get()) / "analysis"
        
        # Enable/disable buttons based on directory existence
        self.view_predictions_button.config(
            state="normal" if predictions_path.exists() else "disabled"
        )
        self.view_analysis_button.config(
            state="normal" if analysis_path.exists() else "disabled"
        )
        
        # Schedule periodic checks
        self.root.after(5000, self.check_results_directories)
    
    def toggle_interactive(self):
        """Enable/disable interactive checkbox based on visualize state"""
        if self.visualize_var.get():
            self.interactive_check.config(state="normal")
        else:
            self.interactive_var.set(False)
            self.interactive_check.config(state="disabled")
    
    def browse_output_dir(self):
        """Open a directory browser to select output directory"""
        directory = filedialog.askdirectory(
            initialdir=self.output_dir_var.get(),
            title="Select Output Directory"
        )
        if directory:
            self.output_dir_var.set(directory)
            self.update_output_status()
    
    def log_message(self, message, message_type="info"):
        """Add a message to the log window with appropriate formatting"""
        self.log_text.config(state="normal")
        
        # Select the appropriate tag based on message content or type
        tag = message_type
        if message_type == "info":
            if message.startswith("ERROR"):
                tag = "error"
            elif message.startswith("WARNING"):
                tag = "warning"
            elif message.startswith("Executing:"):
                tag = "command"
            elif "-----" in message:
                tag = "header"
            elif any(success in message for success in ["complete", "successfully", "✅"]):
                tag = "success"
        
        # Insert the message with the selected tag
        self.log_text.insert(tk.END, message + "\n", tag)
        self.log_text.see(tk.END)  # Scroll to the end
        self.log_text.config(state="disabled")
    
    def clear_logs(self):
        """Clear the log window"""
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")
    
    def save_logs(self):
        """Save log contents to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"lottery_analysis_log_{timestamp}.txt",
            title="Save Log File"
        )
        if filename:
            try:
                with open(filename, "w") as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self.log_message(f"Log saved to {filename}", "success")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save log: {str(e)}")
    
    def confirm_quit(self):
        """Confirm before quitting the application"""
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.root.destroy()
    
    def run_analysis(self):
        """Run the analysis and prediction process"""
        lottery = self.lottery_var.get()
        years = self.years_var.get()
        
        # Clear log if requested
        if messagebox.askyesno("Clear Log", "Do you want to clear the log before starting?"):
            self.clear_logs()
        
        # Update processing state and start time
        self.is_processing = True
        self.start_time = time.time()
        
        # Disable buttons during processing
        self.analyze_button.config(state="disabled")
        
        # Start progress bar
        self.progress.start()
        self.status_var.set(f"Running analysis for {lottery.replace('_', ' ').title()}...")
        
        # Log the start of the process
        self.log_message(f"Starting analysis for {lottery.replace('_', ' ').title()} with {years} years of data")
        self.log_message(f"Generating {self.predictions_var.get()} predictions using the {self.model_var.get()} model")
        self.log_message("Process started at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.log_message("-" * 50, "header")
        
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
                "--format", self.format_var.get(),
                "--output-dir", self.output_dir_var.get()
            ]
            
            # Add optional flags
            if self.visualize_var.get():
                cmd.append("--visualize")
                
                if self.interactive_var.get():
                    cmd.append("--interactive")
            
            if self.report_var.get():
                cmd.append("--output-report")
            
            # Log the command
            self.log_message(f"Executing: {' '.join(cmd)}", "command")
            self.log_message("-" * 50, "header")
            
            # Add network connectivity check
            if not self._check_internet_connection():
                raise ConnectionError("No internet connection available. Required for data scraping.")
            
            # Execute the command
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            # Alert the user about possible browser opening
            self.log_message("NOTE: Firefox browser may open for data scraping. If you see a message about the lottery not being available in your state, quickly switch to NSW.", "warning")
            
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
                self.log_message(f"Process exited with error code {return_code}", "error")
                raise Exception(f"Process exited with error code {return_code}")
            
            # Process completed successfully
            self.status_var.set(f"Analysis complete for {lottery.replace('_', ' ').title()}")
            self.log_message("-" * 50, "header")
            self.log_message(f"Analysis completed successfully!", "success")
            self.log_message(f"Process finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Calculate elapsed time
            elapsed = time.time() - self.start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.log_message(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Update directory checks immediately
            self.check_results_directories()
            
            messagebox.showinfo(
                "Analysis Complete", 
                f"The analysis and prediction for {lottery.replace('_', ' ').title()} is complete.\n\n"
                f"Click 'View Latest Predictions' to see the prediction results or "
                f"'View Analysis Reports' to see the detailed analysis."
            )
            
        except ConnectionError as ce:
            self.status_var.set(f"Connection Error: {str(ce)}")
            self.log_message(f"CONNECTION ERROR: {str(ce)}", "error")
            messagebox.showerror("Connection Error", str(ce))
        except subprocess.SubprocessError as se:
            self.status_var.set(f"Process Error: {str(se)}")
            self.log_message(f"PROCESS ERROR: {str(se)}", "error")
            messagebox.showerror("Process Error", f"Error executing analysis process: {str(se)}")
        except Exception as e:
            error_msg = str(e)
            self.status_var.set(f"Error: {error_msg}")
            self.log_message(f"ERROR: {error_msg}", "error")
            messagebox.showerror("Error", f"An error occurred: {error_msg}")
        
        finally:
            # Reset processing state
            self.is_processing = False
            
            # Stop progress bar and re-enable buttons
            self.progress.stop()
            self.analyze_button.config(state="normal")
    
    def _check_internet_connection(self):
        """Check if internet connection is available"""
        try:
            # Try to connect to a reliable service
            socket.create_connection(("www.google.com", 80), timeout=3)
            return True
        except OSError:
            return False
    
    def view_predictions(self):
        """Open the predictions folder"""
        predictions_path = Path(self.output_dir_var.get()) / "predictions"
        self._open_folder(predictions_path)
    
    def view_analysis(self):
        """Open the analysis folder"""
        analysis_path = Path(self.output_dir_var.get()) / "analysis"
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
            system = platform.system()
            
            if system == "Windows":
                os.startfile(str(path))
            elif system == "Darwin":  # macOS
                subprocess.call(["open", str(path)])
            else:  # Linux
                subprocess.call(["xdg-open", str(path)])
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {str(e)}")

    def setup_settings_tab(self):
        """Set up the settings tab content"""
        settings_frame = ttk.Frame(self.settings_tab, padding="20")
        settings_frame.pack(fill="both", expand=True)
        
        ttk.Label(
            settings_frame,
            text="Application Settings",
            style="Header.TLabel"
        ).pack(pady=(0, 20))
        
        # Theme selection
        theme_frame = ttk.Frame(settings_frame)
        theme_frame.pack(fill="x", pady=10)
        
        ttk.Label(
            theme_frame,
            text="Application Theme:",
            style="Section.TLabel"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        theme_light = ttk.Radiobutton(
            theme_frame,
            text="Light",
            value="light",
            variable=self.theme_var,
            command=lambda: self.apply_theme("light")
        )
        theme_light.pack(side=tk.LEFT, padx=10)
        
        theme_dark = ttk.Radiobutton(
            theme_frame,
            text="Dark",
            value="dark",
            variable=self.theme_var,
            command=lambda: self.apply_theme("dark")
        )
        theme_dark.pack(side=tk.LEFT, padx=10)
        
        # Add a separator
        ttk.Separator(settings_frame, orient="horizontal").pack(fill="x", pady=20)
        
        # Add a save settings button
        save_button = ttk.Button(
            settings_frame,
            text="Save Settings",
            command=self.save_settings
        )
        save_button.pack(pady=10)
        
    def save_settings(self):
        """Save current settings to a configuration file"""
        try:
            settings = {
                "theme": self.theme_var.get(),
                "output_dir": self.output_dir_var.get(),
                "lottery": self.lottery_var.get(),
                "years": self.years_var.get(),
                "predictions": self.predictions_var.get(),
                "model": self.model_var.get(),
                "format": self.format_var.get(),
                "visualize": self.visualize_var.get(),
                "interactive": self.interactive_var.get(),
                "report": self.report_var.get()
            }
            
            # Ensure the directory exists
            os.makedirs("config", exist_ok=True)
            
            # Save to JSON file
            with open("config/settings.json", "w") as f:
                json.dump(settings, f, indent=4)
                
            self.log_message("Settings saved successfully", "success")
            messagebox.showinfo("Settings Saved", "Your settings have been saved successfully.")
            
        except Exception as e:
            self.log_message(f"Error saving settings: {str(e)}", "error")
            messagebox.showerror("Error", f"Could not save settings: {str(e)}")

    def load_settings(self):
        """Load settings from configuration file"""
        try:
            config_path = Path("config/settings.json")
            if config_path.exists():
                with open(config_path, "r") as f:
                    settings = json.load(f)
                    
                # Apply theme first (this affects visuals)
                if "theme" in settings:
                    self.theme_var.set(settings["theme"])
                    self.apply_theme(settings["theme"])  # Actually apply the theme
                    
                # Set output directory
                if "output_dir" in settings:
                    self.output_dir_var.set(settings["output_dir"])
                    self.update_output_status()  # Update the output status display
                    
                # Set lottery type
                if "lottery" in settings:
                    self.lottery_var.set(settings["lottery"])
                    
                # Set years of data
                if "years" in settings and isinstance(settings["years"], int):
                    self.years_var.set(settings["years"])
                    
                # Set prediction count
                if "predictions" in settings and isinstance(settings["predictions"], int):
                    self.predictions_var.set(settings["predictions"])
                    
                # Set model
                if "model" in settings:
                    self.model_var.set(settings["model"])
                    
                # Set format
                if "format" in settings:
                    self.format_var.set(settings["format"])
                    
                # Set visualization
                if "visualize" in settings:
                    self.visualize_var.set(settings["visualize"])
                    
                # Set interactive
                if "interactive" in settings:
                    self.interactive_var.set(settings["interactive"])
                    
                # Apply interactive state based on visualization setting
                self.toggle_interactive()
                    
                # Set report
                if "report" in settings:
                    self.report_var.set(settings["report"])
                    
                # Log successful loading (to console since log might not be ready)
                print(f"Loaded settings from {config_path}")
                self.status_var.set("Settings loaded successfully")
                
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
            # Continue with defaults if loading fails

def main():
    root = tk.Tk()
    app = LotteryAnalyzerGUI(root)
    
    # Handle window close event
    root.protocol("WM_DELETE_WINDOW", app.confirm_quit)
    
    root.mainloop()

if __name__ == "__main__":
    main() 