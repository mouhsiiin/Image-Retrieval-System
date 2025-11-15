"""
Simple GUI for Image Retrieval System

A user-friendly graphical interface for the image retrieval system.
Allows users to:
- Select a folder containing images
- Choose a query image
- View similar images with distances
- Switch between distance metrics

Requirements: tkinter (built-in), PIL
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import json
import cv2
import numpy as np
from pathlib import Path
import threading


class ImageRetrievalGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Retrieval System")
        self.root.geometry("1200x800")
        self.root.configure(bg='white')
        
        # Variables
        self.folder_path = tk.StringVar()
        self.query_image = tk.StringVar()
        self.distance_metric = tk.StringVar(value="euclidean")
        self.top_k = tk.IntVar(value=6)
        
        self.image_files = []
        self.results = []
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets."""
        
        # Configure ttk style for white backgrounds
        style = ttk.Style()
        style.configure('TEntry', fieldbackground='white')
        style.configure('TCombobox', fieldbackground='white', background='white')
        style.map('TCombobox', fieldbackground=[('readonly', 'white')])
        style.map('TCombobox', selectbackground=[('readonly', 'white')])
        
        # =====================================================================
        # TOP FRAME - Controls
        # =====================================================================
        control_frame = tk.Frame(self.root, bg='white')
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Title
        title_label = ttk.Label(
            control_frame, 
            text="🔍 Image Retrieval System",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=5)
        
        # Folder selection
        folder_frame = tk.Frame(control_frame, bg='white')
        folder_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(folder_frame, text="Image Folder:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            folder_frame, 
            textvariable=self.folder_path, 
            width=50
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            folder_frame, 
            text="Browse...", 
            command=self.browse_folder
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            folder_frame, 
            text="Load Images", 
            command=self.load_images
        ).pack(side=tk.LEFT, padx=5)
        
        # Query and settings
        settings_frame = tk.Frame(control_frame, bg='white')
        settings_frame.pack(fill=tk.X, pady=5)
        
        # Query image dropdown
        ttk.Label(settings_frame, text="Query Image:").pack(side=tk.LEFT, padx=5)
        self.query_combo = ttk.Combobox(
            settings_frame, 
            textvariable=self.query_image,
            width=30,
            state="readonly"
        )
        self.query_combo.pack(side=tk.LEFT, padx=5)
        
        # Distance metric
        ttk.Label(settings_frame, text="Distance:").pack(side=tk.LEFT, padx=5)
        distance_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.distance_metric,
            values=["euclidean", "cosine"],
            width=12,
            state="readonly"
        )
        distance_combo.pack(side=tk.LEFT, padx=5)
        
        # Top-K
        ttk.Label(settings_frame, text="Top-K:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(
            settings_frame,
            from_=1,
            to=20,
            textvariable=self.top_k,
            width=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Search button
        ttk.Button(
            settings_frame,
            text="🔍 Search Similar Images",
            command=self.search_similar,
            style="Accent.TButton"
        ).pack(side=tk.LEFT, padx=10)
        
        # Status bar
        self.status_label = ttk.Label(
            control_frame,
            text="Ready. Please select an image folder.",
            relief=tk.SUNKEN
        )
        self.status_label.pack(fill=tk.X, pady=5)
        
        # =====================================================================
        # MAIN FRAME - Results Display
        # =====================================================================
        main_frame = tk.Frame(self.root, bg='white')
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Query image panel
        query_panel = tk.LabelFrame(main_frame, text="Query Image", padx=10, pady=10, bg='white')
        query_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)
        
        self.query_canvas = tk.Canvas(query_panel, width=250, height=250, bg="white", highlightthickness=0)
        self.query_canvas.pack()
        
        self.query_info = tk.Label(query_panel, text="No query selected", wraplength=240, bg='white')
        self.query_info.pack(pady=5)
        
        # Results panel with scrollbar
        results_panel = tk.LabelFrame(main_frame, text="Similar Images", padx=10, pady=10, bg='white')
        results_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Create canvas with scrollbar
        canvas_frame = tk.Frame(results_panel, bg='white')
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            canvas_frame, 
            orient="vertical", 
            command=self.results_canvas.yview
        )
        self.results_frame = tk.Frame(self.results_canvas, bg='white')
        
        self.results_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )
        
        self.results_canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mouse wheel
        self.results_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        self.results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def browse_folder(self):
        """Open folder browser dialog."""
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.folder_path.set(folder)
            self.load_images()
    
    def load_images(self):
        """Load images from the selected folder."""
        folder = self.folder_path.get()
        
        if not folder or not os.path.exists(folder):
            messagebox.showerror("Error", "Please select a valid folder.")
            return
        
        # Get all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif'}
        self.image_files = [
            f for f in os.listdir(folder)
            if Path(f).suffix.lower() in valid_extensions
        ]
        
        if not self.image_files:
            messagebox.showwarning("Warning", "No image files found in the selected folder.")
            return
        
        # Update query combo
        self.query_combo['values'] = sorted(self.image_files)
        if self.image_files:
            self.query_image.set(self.image_files[0])
        
        self.status_label.config(
            text=f"Loaded {len(self.image_files)} images from {os.path.basename(folder)}"
        )
        
        # Clear previous results
        self.clear_results()
    
    def clear_results(self):
        """Clear all result displays."""
        # Clear query display
        self.query_canvas.delete("all")
        self.query_info.config(text="No query selected")
        
        # Clear results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
    
    def load_image_for_display(self, image_path, max_size=(200, 200)):
        """Load and resize image for display."""
        try:
            img = Image.open(image_path)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def display_query_image(self, image_path):
        """Display the query image."""
        self.query_canvas.delete("all")
        
        photo = self.load_image_for_display(image_path, (240, 240))
        if photo:
            # Keep a reference to prevent garbage collection
            self.query_photo = photo
            
            # Center image on canvas
            self.query_canvas.create_image(125, 125, image=photo)
            
            # Update info
            self.query_info.config(text=f"Query: {os.path.basename(image_path)}")
    
    def search_similar(self):
        """Search for similar images."""
        folder = self.folder_path.get()
        query = self.query_image.get()
        
        if not folder or not query:
            messagebox.showwarning("Warning", "Please select a folder and query image.")
            return
        
        # Check if features exist
        query_json = Path(query).stem + '.json'
        query_json_path = os.path.join(folder, query_json)
        
        if not os.path.exists(query_json_path):
            response = messagebox.askyesno(
                "Features Not Found",
                "Features have not been extracted for this folder.\n\n"
                "Would you like to extract features now?\n"
                "(This may take a few minutes)"
            )
            if response:
                self.extract_features(folder)
            return
        
        # Run search in background thread
        self.status_label.config(text="Searching for similar images...")
        self.root.update()
        
        thread = threading.Thread(target=self._search_thread, args=(folder, query))
        thread.start()
    
    def _search_thread(self, folder, query):
        """Run search in background thread."""
        try:
            from b_image_search import find_similar_images
            
            results = find_similar_images(
                query,
                folder,
                self.top_k.get(),
                self.distance_metric.get()
            )
            
            # Update UI in main thread
            self.root.after(0, lambda: self.display_results(folder, query, results))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Search failed:\n{str(e)}"))
            self.root.after(0, lambda: self.status_label.config(text="Search failed."))
    
    def display_results(self, folder, query, results):
        """Display search results."""
        # Display query image
        query_path = os.path.join(folder, query)
        self.display_query_image(query_path)
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Display results in a grid
        cols = 3
        self.result_photos = []  # Keep references
        
        for idx, (image_name, distance) in enumerate(results):
            row = idx // cols
            col = idx % cols
            
            # Create frame for each result
            result_frame = tk.Frame(self.results_frame, relief=tk.RIDGE, borderwidth=2, bg='white')
            result_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            # Load and display image
            image_path = os.path.join(folder, image_name)
            photo = self.load_image_for_display(image_path, (180, 180))
            
            if photo:
                self.result_photos.append(photo)
                
                # Image label
                img_label = tk.Label(result_frame, image=photo, bg='white')
                img_label.pack(pady=5)
                
                # Rank badge
                rank_colors = ['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen', 'lavender']
                rank_color = rank_colors[idx] if idx < len(rank_colors) else 'lightgray'
                
                rank_label = tk.Label(
                    result_frame,
                    text=f"Rank #{idx + 1}",
                    bg=rank_color,
                    font=("Arial", 10, "bold")
                )
                rank_label.pack()
                
                # Image name
                name_label = tk.Label(
                    result_frame,
                    text=image_name,
                    wraplength=170,
                    font=("Arial", 9),
                    bg='white'
                )
                name_label.pack()
                
                # Distance
                dist_label = tk.Label(
                    result_frame,
                    text=f"Distance: {distance:.4f}",
                    font=("Arial", 9, "italic"),
                    bg='white'
                )
                dist_label.pack(pady=2)
        
        # Configure grid weights
        for i in range(cols):
            self.results_frame.columnconfigure(i, weight=1)
        
        # Update status
        self.status_label.config(
            text=f"Found {len(results)} similar images using {self.distance_metric.get()} distance"
        )
    
    def extract_features(self, folder):
        """Extract features from all images in folder."""
        response = messagebox.showinfo(
            "Extract Features",
            f"This will extract features from all images in:\n{folder}\n\n"
            "Please run this command in a terminal:\n\n"
            f"python a_extract_features.py --folder \"{folder}\"\n\n"
            "Then try searching again.",
            icon='info'
        )


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    
    # Set theme (if available)
    try:
        style = ttk.Style()
        style.theme_use('clam')  # or 'alt', 'default', 'classic'
    except:
        pass
    
    # Create and run app
    app = ImageRetrievalGUI(root)
    
    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    print("Starting Image Retrieval System GUI...")
    print("=" * 70)
    print("Make sure you have extracted features first using:")
    print("  python a_extract_features.py --folder <folder_name>")
    print("=" * 70)
    main()
