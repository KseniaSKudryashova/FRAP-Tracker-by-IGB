# =============================================================================
#                     🧪 FRAP-Tracker BASIC by IGB 🎯
#               Fixed FOCI relative to nucleus - V2026.03.09 
# =============================================================================

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import numpy as np
import pandas as pd
from skimage import io, measure, morphology, feature, filters, segmentation, draw
from scipy import ndimage
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle
import os
import datetime
from PIL import Image, ImageDraw, ImageFont


class FRAPTrackerBasic:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.total_roi_points = None
        self.background_roi_points = None
        self.reference_nucleus_mask = None
        self.reference_center = None
        self.reference_area = None
        self.tracked_results = {}
        self.foci_roi_radius = None
        self.foci_roi_center_bleach = None
        self.nucleus_center_bleach = None
        self.relative_foci_position = None
        self.processed_frames = []
        self.user_selected_foci_radius = None
        self.bleach_frame_original_center = None
        self.bleach_frame_gray = None  # For tracking
        self.bleach_frame_processed = False  # Flag that BLEACH frame is processed
        self.current_nucleus_mask = None  # Current nucleus mask for tracking
        self.first_pre_frame_gray = None  # First PRE frame for tracking
        self.frame_interval = None  # Interval between frames in seconds
        self.experiment_name = None  # User-defined experiment name

    def load_image_files(self):
        print("📁 Loading image files...")

        image_files = {}
        file_types = [
            ("PRE_BLEACH", "PRE BLEACH", "Select PRE BLEACH (TIFF)"),
            ("BLEACH", "BLEACH", "Select BLEACH (TIFF)"),
            ("POST_BLEACH", "POST BLEACH", "Select POST BLEACH (TIFF)"),
        ]

        for file_key, file_desc, dialog_title in file_types:
            while True:
                messagebox.showinfo(f"Select {file_desc}", dialog_title)
                file_path = filedialog.askopenfilename(
                    title=dialog_title,
                    filetypes=[("TIFF files", "*.tif;*.tiff"), ("All files", "*.*")],
                )

                if not file_path:
                    if messagebox.askretrycancel(
                        "File not selected", "Try again?"
                    ):
                        continue
                    else:
                        return None
                else:
                    if os.path.exists(file_path):
                        image_files[file_key] = file_path
                        print(f"✅ {file_desc}: {os.path.basename(file_path)}")
                        break
                    else:
                        messagebox.showerror(
                            "Error", f"File does not exist: {file_path}"
                        )

        return image_files

    def ask_experiment_name(self):
        """Ask user for experiment name for output file"""
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title("Experiment Name")
            dialog.geometry("450x200")
            dialog.attributes("-topmost", True)
            
            # Center the window
            dialog.update_idletasks()
            width = dialog.winfo_width()
            height = dialog.winfo_height()
            x = (dialog.winfo_screenwidth() // 2) - (width // 2)
            y = (dialog.winfo_screenheight() // 2) - (height // 2)
            dialog.geometry(f'{width}x{height}+{x}+{y}')
            
            tk.Label(dialog, text="EXPERIMENT IDENTIFICATION", 
                    font=("Arial", 14, "bold")).pack(pady=15)
            
            tk.Label(dialog, text="Enter experiment name for output file:",
                    font=("Arial", 11)).pack(pady=10)
            
            tk.Label(dialog, text="(will be used as: forEasyFRAP_experimentname.csv)",
                    font=("Arial", 9), fg="gray").pack()
            
            frame = tk.Frame(dialog)
            frame.pack(pady=10)
            
            name_var = tk.StringVar(value="experiment")
            entry = tk.Entry(frame, textvariable=name_var, width=30, 
                           font=("Arial", 11), justify="center")
            entry.pack(side=tk.LEFT, padx=5)
            
            result = None
            
            def on_ok():
                nonlocal result
                name = name_var.get().strip()
                if not name:
                    messagebox.showerror("Error", "Please enter a name!")
                    return
                # Remove any invalid characters
                name = "".join(c for c in name if c.isalnum() or c in "_-").rstrip()
                if not name:
                    name = "experiment"
                result = name
                dialog.destroy()
            
            def on_cancel():
                nonlocal result
                result = "experiment"  # Default if cancelled
                dialog.destroy()
            
            # Buttons
            button_frame = tk.Frame(dialog)
            button_frame.pack(pady=20)
            
            tk.Button(button_frame, text="✅ OK", command=on_ok, 
                     bg="lightgreen", font=("Arial", 10), width=12,
                     padx=10, pady=5).pack(side=tk.LEFT, padx=10)
            tk.Button(button_frame, text="❌ CANCEL", command=on_cancel,
                     bg="lightcoral", font=("Arial", 10), width=12,
                     padx=10, pady=5).pack(side=tk.LEFT, padx=10)
            
            # Focus on input field
            entry.focus_set()
            entry.select_range(0, tk.END)
            
            # Bind Enter key
            dialog.bind('<Return>', lambda e: on_ok())
            
            dialog.wait_window()
            
            print(f"✅ Experiment name set: {result}")
            return result
                
        except Exception as e:
            print(f"❌ Error in experiment name input: {e}")
            return "experiment"

    def ask_frame_interval(self):
        """Ask user for frame interval (in seconds)"""
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title("Frame Interval")
            dialog.geometry("400x200")
            dialog.attributes("-topmost", True)
            
            # Center the window
            dialog.update_idletasks()
            width = dialog.winfo_width()
            height = dialog.winfo_height()
            x = (dialog.winfo_screenwidth() // 2) - (width // 2)
            y = (dialog.winfo_screenheight() // 2) - (height // 2)
            dialog.geometry(f'{width}x{height}+{x}+{y}')
            
            tk.Label(dialog, text="FRAP TIME PARAMETERS", 
                    font=("Arial", 14, "bold")).pack(pady=15)
            
            tk.Label(dialog, text="Enter frame interval (in seconds):",
                    font=("Arial", 11)).pack(pady=10)
            
            frame = tk.Frame(dialog)
            frame.pack(pady=10)
            
            tk.Label(frame, text="Interval:").pack(side=tk.LEFT, padx=5)
            
            interval_var = tk.StringVar(value="5.0")
            entry = tk.Entry(frame, textvariable=interval_var, width=10, 
                           font=("Arial", 11), justify="center")
            entry.pack(side=tk.LEFT, padx=5)
            
            tk.Label(frame, text="seconds").pack(side=tk.LEFT, padx=5)
            
            result = None
            
            def on_ok():
                nonlocal result
                try:
                    interval = float(interval_var.get())
                    if interval <= 0:
                        messagebox.showerror("Error", "Interval must be positive!")
                        return
                    result = interval
                    dialog.destroy()
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid number!")
            
            def on_cancel():
                nonlocal result
                result = None
                dialog.destroy()
            
            # Buttons
            button_frame = tk.Frame(dialog)
            button_frame.pack(pady=20)
            
            tk.Button(button_frame, text="✅ OK", command=on_ok, 
                     bg="lightgreen", font=("Arial", 10), width=12,
                     padx=10, pady=5).pack(side=tk.LEFT, padx=10)
            tk.Button(button_frame, text="❌ CANCEL", command=on_cancel,
                     bg="lightcoral", font=("Arial", 10), width=12,
                     padx=10, pady=5).pack(side=tk.LEFT, padx=10)
            
            # Focus on input field
            entry.focus_set()
            entry.select_range(0, tk.END)
            
            # Bind Enter key
            dialog.bind('<Return>', lambda e: on_ok())
            
            dialog.wait_window()
            
            if result is not None:
                print(f"✅ Frame interval set: {result} seconds")
                return result
            else:
                print("❌ User cancelled interval input")
                return None
                
        except Exception as e:
            print(f"❌ Error in interval input: {e}")
            return None

    def select_all_rois_on_bleach(self, bleach_image):
        print("🎯 SEQUENTIAL ROI SELECTION ON BLEACH...")

        if not self.select_foci_roi_on_bleach(bleach_image):
            print("❌ FOCI ROI selection cancelled")
            return False

        if not self.select_nucleus_roi_on_bleach(bleach_image):
            print("❌ NUCLEUS ROI selection cancelled")
            return False

        self.calculate_relative_foci_position()
        return True

    def select_foci_roi_on_bleach(self, bleach_image):
        print("🎯 FOCI ROI SELECTION ON BLEACH...")

        class FociROISelectionState:
            def __init__(self, image):
                self.image = image
                self.center = None
                self.radius = None
                self.finished = False

        if len(bleach_image.shape) == 3:
            bleach_frame = (
                bleach_image[0]
                if bleach_image.shape[0] in [1, 3]
                else bleach_image.mean(axis=0)
            )
        else:
            bleach_frame = bleach_image

        state = FociROISelectionState(bleach_frame)

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)

        display_image = self.optimize_contrast_for_display(state.image)

        def update_display():
            ax.clear()
            ax.imshow(display_image, cmap="gray")

            if state.center and state.radius:
                circle = Circle(
                    state.center, state.radius, fill=False, color="red", linewidth=2
                )
                ax.add_patch(circle)
                ax.plot(
                    state.center[0],
                    state.center[1],
                    "r+",
                    markersize=15,
                    markeredgewidth=2,
                )
                ax.text(
                    state.center[0] + state.radius + 5,
                    state.center[1],
                    f"R: {state.radius:.1f}px",
                    color="red",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_title("FOCI ROI SELECTION: click for center")
            ax.axis("off")
            plt.draw()

        def on_click(event):
            if event.inaxes == ax and event.button == 1:
                state.center = (float(event.xdata), float(event.ydata))
                print(f"✅ FOCI center: ({event.xdata:.1f}, {event.ydata:.1f})")
                update_display()

        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        radius_slider = Slider(
            ax_slider, "FOCI radius", 1, 20, valinit=5, valfmt="%.0f", valstep=1
        )

        ax_finish = plt.axes([0.3, 0.05, 0.2, 0.04])
        ax_cancel = plt.axes([0.55, 0.05, 0.15, 0.04])

        btn_finish = Button(ax_finish, "✅ Finish", color="lightgreen")
        btn_cancel = Button(ax_cancel, "❌ Cancel", color="lightcoral")

        def on_slider_change(val):
            state.radius = int(val)
            update_display()

        def finish_selection(event):
            if not state.center or not state.radius:
                messagebox.showwarning("Incomplete", "Select FOCI center and radius!")
                return
            state.finished = True
            plt.close()
            print(f"✅ FOCI ROI: center {state.center}, radius {state.radius:.1f}px")

        def cancel_selection(event):
            state.finished = False
            plt.close()
            print("❌ FOCI ROI selection cancelled")

        state.radius = int(radius_slider.val)
        radius_slider.on_changed(on_slider_change)
        btn_finish.on_clicked(finish_selection)
        btn_cancel.on_clicked(cancel_selection)
        fig.canvas.mpl_connect("button_press_event", on_click)
        update_display()
        plt.show()

        if state.finished:
            user_selected_radius = float(state.radius)
            self.foci_roi_center_bleach = (
                float(state.center[0]),
                float(state.center[1]),
            )
            self.foci_roi_radius = user_selected_radius
            self.user_selected_foci_radius = user_selected_radius
            self.bleach_frame_original_center = (
                float(state.center[0]),
                float(state.center[1]),
            )

            print(
                f"✅ FOCI saved: center {self.foci_roi_center_bleach}, radius {self.foci_roi_radius}px"
            )
            return True

        return False

    def select_nucleus_roi_on_bleach(self, bleach_image):
        print("🎯 NUCLEUS ROI SELECTION ON BLEACH...")

        class NucleusSelectionState:
            def __init__(self, image):
                self.image = image
                self.points = []
                self.finished = False

        if len(bleach_image.shape) == 3:
            bleach_frame = (
                bleach_image[0]
                if bleach_image.shape[0] in [1, 3]
                else bleach_image.mean(axis=0)
            )
        else:
            bleach_frame = bleach_image

        state = NucleusSelectionState(bleach_frame)

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.15)
        display_image = self.optimize_contrast_for_display(state.image)

        def update_display():
            ax.clear()
            ax.imshow(display_image, cmap="gray")

            if state.points:
                ax.plot(
                    [p[0] for p in state.points],
                    [p[1] for p in state.points],
                    "go",
                    markersize=6,
                )
                if len(state.points) > 1:
                    points_array = np.array(state.points)
                    ax.plot(
                        points_array[:, 0],
                        points_array[:, 1],
                        "g-",
                        linewidth=2,
                        alpha=0.7,
                    )

            ax.set_title("NUCLEUS ROI SELECTION: Trace nucleus outline")
            ax.text(
                0.02,
                0.98,
                f"Points: {len(state.points)}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="green", alpha=0.8),
            )
            ax.axis("off")
            plt.draw()

        def on_click(event):
            if event.inaxes == ax and event.button == 1:
                state.points.append([float(event.xdata), float(event.ydata)])
                print(f"✅ Nucleus point: ({event.xdata:.1f}, {event.ydata:.1f})")
                update_display()

        ax_remove_point = plt.axes([0.3, 0.05, 0.2, 0.05])
        ax_finish_selection = plt.axes([0.6, 0.05, 0.3, 0.05])

        btn_remove_point = Button(
            ax_remove_point, "❌ Remove point", color="lightcoral"
        )
        btn_finish_selection = Button(
            ax_finish_selection, "✅ FINISH", color="lightgreen"
        )

        def remove_point(event):
            if state.points:
                removed_point = state.points.pop()
                print(f"❌ Point removed")
                update_display()

        def finish_selection(event):
            if len(state.points) < 5:
                messagebox.showwarning("Too few points", "Minimum 5 points required!")
                return

            if state.points[0] != state.points[-1]:
                state.points.append(state.points[0])

            state.finished = True
            plt.close()
            print(f"✅ Nucleus selection: {len(state.points)} points")

        btn_remove_point.on_clicked(remove_point)
        btn_finish_selection.on_clicked(finish_selection)
        fig.canvas.mpl_connect("button_press_event", on_click)
        update_display()
        plt.show()

        if state.finished:
            from skimage.draw import polygon

            mask = np.zeros(display_image.shape, dtype=bool)

            x_coords = [p[0] for p in state.points]
            y_coords = [p[1] for p in state.points]
            rr, cc = polygon(y_coords, x_coords, display_image.shape)
            mask[rr, cc] = True

            self.reference_nucleus_mask = mask
            # Use GEOMETRIC center instead of intensity center
            self.nucleus_center_bleach = self.get_geometric_center(mask)
            self.reference_area = np.sum(mask)

            # Save current mask for tracking
            self.current_nucleus_mask = mask

            print(
                f"✅ Nucleus: center {self.nucleus_center_bleach}, area {self.reference_area}px"
            )
            return True
        return False

    def calculate_relative_foci_position(self):
        """Calculate relative position of FOCI relative to nucleus"""
        if self.foci_roi_center_bleach is None or self.nucleus_center_bleach is None:
            print("❌ Cannot calculate relative position")
            return False

        foci_center = np.array(self.foci_roi_center_bleach, dtype=float)
        nucleus_center = np.array(self.nucleus_center_bleach, dtype=float)

        self.relative_foci_position = foci_center - nucleus_center

        print(f"✅ Relative position: {self.relative_foci_position}")
        return True

    def get_foci_position_from_nucleus(self, nucleus_center):
        """Get FOCI position based on nucleus position"""
        if nucleus_center is None or self.relative_foci_position is None:
            print("❌ Cannot determine FOCI position")
            return None

        foci_position = np.array(nucleus_center, dtype=float) + np.array(
            self.relative_foci_position, dtype=float
        )
        return foci_position

    def get_geometric_center(self, mask):
        """Get geometric center of binary mask (intensity independent)"""
        if mask is None or np.sum(mask) == 0:
            print("⚠️ Mask is empty or None")
            return None

        # Find coordinates of all mask pixels
        coords = np.argwhere(mask)

        if len(coords) == 0:
            return None

        # Calculate mean of X and Y coordinates
        # coords[:, 1] - X coordinates (columns)
        # coords[:, 0] - Y coordinates (rows)
        center_x = np.mean(coords[:, 1])
        center_y = np.mean(coords[:, 0])

        return np.array([center_x, center_y])

    def track_nucleus_across_frames(
        self, reference_frame, target_frame, reference_mask, direction="forward"
    ):
        """Track nucleus between frames using Phase Correlation (more reliable method)"""
        try:
            # Convert to grayscale
            if len(reference_frame.shape) == 3:
                reference_frame_gray = reference_frame.mean(axis=2)
            else:
                reference_frame_gray = reference_frame
                
            if len(target_frame.shape) == 3:
                target_frame_gray = target_frame.mean(axis=2)
            else:
                target_frame_gray = target_frame
                
            # Use GEOMETRIC center
            reference_center_xy = self.get_geometric_center(reference_mask)
            
            if reference_center_xy is None:
                print("⚠️ Cannot compute geometric center of mask")
                reference_center_xy = np.array(reference_frame_gray.shape[::-1]) / 2
                
            # Method 1: Phase Correlation (more reliable for displacements)
            try:
                # Apply Gaussian filter to reduce noise
                ref_filtered = filters.gaussian(reference_frame_gray, sigma=1)
                target_filtered = filters.gaussian(target_frame_gray, sigma=1)
                
                # Compute phase correlation
                ref_fft = np.fft.fft2(ref_filtered)
                target_fft = np.fft.fft2(target_filtered)
                
                # Compute cross-power spectrum
                cross_power = ref_fft * np.conj(target_fft)
                cross_power /= np.abs(cross_power) + 1e-10  # for stability
                
                # Inverse Fourier transform
                correlation = np.fft.ifft2(cross_power).real
                
                # Find maximum correlation
                max_pos = np.unravel_index(np.argmax(correlation), correlation.shape)
                
                # Calculate displacement
                rows, cols = correlation.shape
                displacement_y = (max_pos[0] + rows//2) % rows - rows//2
                displacement_x = (max_pos[1] + cols//2) % cols - cols//2
                
                displacement_xy = np.array([displacement_x, displacement_y])
                
            except Exception as e1:
                # Method 2: Enhanced cross-correlation
                try:
                    correlation = feature.match_template(
                        target_frame_gray, reference_frame_gray, pad_input=True
                    )
                    max_pos = np.unravel_index(np.argmax(correlation), correlation.shape)
                    
                    # Adjust displacement
                    rows, cols = reference_frame_gray.shape
                    displacement_y = max_pos[0] - rows
                    displacement_x = max_pos[1] - cols
                    
                    displacement_xy = np.array([displacement_x, displacement_y])
                    
                except Exception as e2:
                    displacement_xy = np.array([0, 0])
            
            # New center = old center + displacement
            new_center_xy = reference_center_xy + displacement_xy
            
            # Create new mask based on displacement
            if reference_mask is not None:
                from scipy.ndimage import shift
                
                # Convert displacement for shift (Y, X order)
                displacement_yx = np.array([displacement_xy[1], displacement_xy[0]])
                
                tracked_mask = shift(
                    reference_mask.astype(float),
                    displacement_yx,
                    order=0,
                    mode='nearest'
                ) > 0.5
                
                # Check if mask is not empty after shift
                if np.sum(tracked_mask) == 0:
                    tracked_mask = reference_mask
            else:
                tracked_mask = reference_mask
                
            return tracked_mask, new_center_xy
            
        except Exception as e:
            print(f"❌ Error in nucleus tracking: {e}")
            
            if reference_mask is not None:
                return reference_mask, self.get_geometric_center(reference_mask)
            else:
                return None, np.array([0, 0])

    def create_nucleus_roi_from_mask(self, nucleus_mask):
        if nucleus_mask is None:
            return None

        contours = measure.find_contours(nucleus_mask, 0.5)

        if not contours:
            print("❌ No nucleus contour found")
            return None

        main_contour = max(contours, key=len)
        simplified_contour = main_contour[::3]

        roi_points = simplified_contour[:, [1, 0]].tolist()

        if roi_points[0] != roi_points[-1]:
            roi_points.append(roi_points[0])

        return roi_points

    def optimize_contrast_for_display(self, image):
        p2, p98 = np.percentile(image, (2, 98))
        if p98 - p2 > 0:
            normalized = np.clip((image - p2) / (p98 - p2), 0, 1)
        else:
            normalized = image / image.max() if image.max() > 0 else image
        return normalized

    def measure_intensity_in_roi(self, image, center, radius):
        """Measure intensity in circular ROI"""
        if center is None:
            return 0

        center_x, center_y = float(center[0]), float(center[1])
        y_indices, x_indices = np.ogrid[: image.shape[0], : image.shape[1]]
        mask = ((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2) <= radius**2

        if np.sum(mask) == 0:
            return 0
        return np.mean(image[mask])

    def measure_intensity_in_polygon_roi(self, image, roi_points):
        if not roi_points or len(roi_points) < 3:
            return 0

        from skimage.draw import polygon

        roi_mask = np.zeros(image.shape[:2], dtype=bool)
        x_coords = [p[0] for p in roi_points]
        y_coords = [p[1] for p in roi_points]
        rr, cc = polygon(y_coords, x_coords, image.shape[:2])
        roi_mask[rr, cc] = True

        if len(image.shape) == 3:
            image = image.mean(axis=2)

        return np.mean(image[roi_mask])

    def measure_background_intensity_roi(self, image, background_points):
        if not background_points or len(background_points) < 3:
            return 0

        from skimage.draw import polygon

        roi_mask = np.zeros(image.shape[:2], dtype=bool)
        x_coords = [p[0] for p in background_points]
        y_coords = [p[1] for p in background_points]
        rr, cc = polygon(y_coords, x_coords, image.shape[:2])
        roi_mask[rr, cc] = True

        if len(image.shape) == 3:
            image = image.mean(axis=2)

        return np.mean(image[roi_mask])

    def create_frame_with_outlines(
        self, frame, foci_center, nucleus_center, nucleus_roi, frame_id, time_point
    ):
        """Create frame with outlines"""
        try:
            # Convert to RGB if needed
            if len(frame.shape) == 2:
                display_frame = np.stack([frame] * 3, axis=-1)
            else:
                display_frame = frame.copy()
                if display_frame.shape[2] == 1:
                    display_frame = np.stack([display_frame[:, :, 0]] * 3, axis=-1)

            # Normalize for display
            display_frame = display_frame.astype(np.float32)
            if display_frame.max() > display_frame.min():
                display_frame = (display_frame - display_frame.min()) / (
                    display_frame.max() - display_frame.min()
                )

            # Draw FOCI (red circle and yellow cross)
            if foci_center is not None and self.foci_roi_radius is not None:
                foci_x, foci_y = float(foci_center[0]), float(foci_center[1])
                radius_to_use = (
                    self.user_selected_foci_radius
                    if self.user_selected_foci_radius is not None
                    else self.foci_roi_radius
                )
                radius = int(radius_to_use)

                # FOCI circle (red)
                rr, cc = draw.circle_perimeter(
                    int(foci_y), int(foci_x), radius, shape=frame.shape[:2]
                )
                valid = (
                    (rr >= 0)
                    & (rr < frame.shape[0])
                    & (cc >= 0)
                    & (cc < frame.shape[1])
                )
                display_frame[rr[valid], cc[valid], :] = [1.0, 0.0, 0.0]

                # Cross at FOCI center (yellow)
                for dy in range(-5, 6):
                    y_pos = int(foci_y) + dy
                    if 0 <= y_pos < frame.shape[0]:
                        display_frame[y_pos, int(foci_x), :] = [1, 1, 0]
                for dx in range(-5, 6):
                    x_pos = int(foci_x) + dx
                    if 0 <= x_pos < frame.shape[1]:
                        display_frame[int(foci_y), x_pos, :] = [1, 1, 0]

            # Draw NUCLEUS ROI (green)
            if nucleus_roi is not None and len(nucleus_roi) > 2:
                points = np.array(nucleus_roi)
                for i in range(len(points) - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    rr, cc = draw.line(int(y1), int(x1), int(y2), int(x2))
                    valid = (
                        (rr >= 0)
                        & (rr < frame.shape[0])
                        & (cc >= 0)
                        & (cc < frame.shape[1])
                    )
                    display_frame[rr[valid], cc[valid], :] = [0.0, 1.0, 0.0]

            # Draw nucleus center (larger green dot)
            if nucleus_center is not None:
                nuc_x, nuc_y = float(nucleus_center[0]), float(nucleus_center[1])
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        y_pos = int(nuc_y) + dy
                        x_pos = int(nuc_x) + dx
                        if 0 <= y_pos < frame.shape[0] and 0 <= x_pos < frame.shape[1]:
                            display_frame[y_pos, x_pos, :] = [0, 1, 0]

            # Draw BACKGROUND (blue)
            if (
                self.background_roi_points is not None
                and len(self.background_roi_points) > 2
            ):
                points = np.array(self.background_roi_points)
                for i in range(len(points) - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    rr, cc = draw.line(int(y1), int(x1), int(y2), int(x2))
                    valid = (
                        (rr >= 0)
                        & (rr < frame.shape[0])
                        & (cc >= 0)
                        & (cc < frame.shape[1])
                    )
                    display_frame[rr[valid], cc[valid], :] = [0.0, 0.0, 1.0]

            # Convert to uint8
            display_frame = (display_frame * 255).astype(np.uint8)
            
            # Create PIL image
            pil_image = Image.fromarray(display_frame)
            draw_obj = ImageDraw.Draw(pil_image)

            # Font
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()

            # Text information
            info_text = f"{frame_id}\nTime: {time_point:.1f}s"
            draw_obj.text((10, 10), info_text, fill=(255, 255, 0), font=font)

            return np.array(pil_image)

        except Exception as e:
            print(f"❌ Error creating frame: {e}")
            return frame

    def select_background_roi_with_slider(
        self, pre_bleach_image, bleach_image, post_bleach_image
    ):
        print("🔄 Starting BACKGROUND ROI selection with slider across all frames...")

        all_frames = []
        frame_info = []

        if len(pre_bleach_image.shape) == 3:
            for i in range(pre_bleach_image.shape[0]):
                all_frames.append(pre_bleach_image[i])
                frame_info.append(f"PRE_{i}")
        else:
            all_frames.append(pre_bleach_image)
            frame_info.append("PRE_0")

        if len(bleach_image.shape) == 3:
            for i in range(bleach_image.shape[0]):
                all_frames.append(bleach_image[i])
                frame_info.append(f"BLEACH_{i}")
        else:
            all_frames.append(bleach_image)
            frame_info.append("BLEACH_0")

        if len(post_bleach_image.shape) == 3:
            for i in range(post_bleach_image.shape[0]):
                all_frames.append(post_bleach_image[i])
                frame_info.append(f"POST_{i}")
        else:
            all_frames.append(post_bleach_image)
            frame_info.append("POST_0")

        class BackgroundROIState:
            def __init__(self, stack, info):
                self.stack = stack
                self.frame_info = info
                self.current_frame_idx = 0
                self.points = []
                self.finished = False

        state = BackgroundROIState(all_frames, frame_info)

        fig, ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.25)

        def show_current_frame():
            ax.clear()
            current_frame = state.stack[state.current_frame_idx]

            if len(current_frame.shape) == 3:
                display_frame = (
                    current_frame[0]
                    if current_frame.shape[0] in [1, 3]
                    else current_frame.mean(axis=0)
                )
            else:
                display_frame = current_frame

            display_frame = self.optimize_contrast_for_display(display_frame)
            ax.imshow(display_frame, cmap="gray")

            if state.points:
                ax.plot(
                    [p[0] for p in state.points],
                    [p[1] for p in state.points],
                    "bo",
                    markersize=6,
                )

                if len(state.points) > 1:
                    points_array = np.array(state.points)
                    ax.plot(points_array[:, 0], points_array[:, 1], "b-", linewidth=2)

                if len(state.points) > 2:
                    closed_points = state.points + [state.points[0]]
                    closed_array = np.array(closed_points)
                    ax.plot(
                        closed_array[:, 0],
                        closed_array[:, 1],
                        "b--",
                        linewidth=1,
                        alpha=0.7,
                    )

            frame_type = state.frame_info[state.current_frame_idx]
            ax.set_title(f"BACKGROUND ROI - {frame_type}\nSelect area WITHOUT cells")
            ax.text(
                0.02,
                0.98,
                f"Points: {len(state.points)}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="blue", alpha=0.8),
            )

            plt.draw()

        ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
        slider = Slider(
            ax_slider, "Frame", 1, len(state.stack), valinit=1, valfmt="%d", valstep=1
        )

        ax_remove_point = plt.axes([0.3, 0.05, 0.15, 0.05])
        ax_finish_roi = plt.axes([0.5, 0.05, 0.2, 0.05])
        ax_cancel = plt.axes([0.75, 0.05, 0.15, 0.05])

        btn_remove_point = Button(
            ax_remove_point, "❌ Remove point", color="lightcoral"
        )
        btn_finish_roi = Button(ax_finish_roi, "✅ Finish selection", color="lightgreen")
        btn_cancel = Button(ax_cancel, "❌ Cancel", color="lightcoral")

        def on_slider_change(val):
            state.current_frame_idx = int(val) - 1
            show_current_frame()

        def on_click(event):
            if event.inaxes == ax and event.button == 1:
                state.points.append([float(event.xdata), float(event.ydata)])
                current_frame_type = state.frame_info[state.current_frame_idx]
                print(
                    f"✅ Added BACKGROUND point on {current_frame_type}: ({event.xdata:.1f}, {event.ydata:.1f})"
                )
                show_current_frame()

        def remove_point(event):
            if state.points:
                removed_point = state.points.pop()
                current_frame_type = state.frame_info[state.current_frame_idx]
                print(
                    f"❌ Removed BACKGROUND point from {current_frame_type}: ({removed_point[0]:.1f}, {removed_point[1]:.1f})"
                )
                show_current_frame()

        def finish_roi(event):
            if len(state.points) < 3:
                messagebox.showwarning("Too few points", "Minimum 3 points for ROI!")
                return

            if state.points[0] != state.points[-1]:
                state.points.append(state.points[0])

            state.finished = True
            plt.close()
            print(f"✅ BACKGROUND ROI completed: {len(state.points)} points")

        def cancel_selection(event):
            state.points = []
            state.finished = False
            plt.close()
            print("❌ BACKGROUND ROI selection cancelled")

        slider.on_changed(on_slider_change)
        btn_remove_point.on_clicked(remove_point)
        btn_finish_roi.on_clicked(finish_roi)
        btn_cancel.on_clicked(cancel_selection)
        fig.canvas.mpl_connect("button_press_event", on_click)

        show_current_frame()

        print("\n🖱️ ENHANCED BACKGROUND ROI SELECTION:")
        print("1. Use slider to navigate through ALL frames (PRE-BLEACH-POST)")
        print("2. Click LMB to add BACKGROUND ROI points")
        print("3. Select area WITHOUT cells for background measurement")
        print("4. Click '✅ Finish selection' when done")

        plt.show()

        if state.finished:
            self.background_roi_points = state.points
            return True
        else:
            self.background_roi_points = None
            return False

    def process_bleach_frame(self, bleach_frame, current_time):
        """Process BLEACH frame - use original coordinates"""
        if len(bleach_frame.shape) == 3:
            frame_gray = (
                bleach_frame[0]
                if bleach_frame.shape[0] in [1, 3]
                else bleach_frame.mean(axis=0)
            )
        else:
            frame_gray = bleach_frame

        # Save BLEACH frame for tracking
        self.bleach_frame_gray = frame_gray

        # For BLEACH frame use ORIGINAL coordinates
        foci_center = self.foci_roi_center_bleach
        nucleus_center = self.nucleus_center_bleach
        self.total_roi_points = self.create_nucleus_roi_from_mask(
            self.reference_nucleus_mask
        )

        # Measure intensities
        if foci_center is not None:
            radius_to_use = (
                self.user_selected_foci_radius
                if self.user_selected_foci_radius is not None
                else self.foci_roi_radius
            )
            foci_intensity = self.measure_intensity_in_roi(
                frame_gray, foci_center, radius_to_use
            )
        else:
            foci_intensity = 0

        nucleus_intensity = (
            self.measure_intensity_in_polygon_roi(frame_gray, self.total_roi_points)
            if self.total_roi_points
            else 0
        )
        background_intensity = (
            self.measure_background_intensity_roi(
                frame_gray, self.background_roi_points
            )
            if self.background_roi_points
            else 0
        )

        print(f"📊 BLEACH: FOCI={foci_intensity:.1f}, NUCLEUS={nucleus_intensity:.1f}, BG={background_intensity:.1f}")

        # Create frame with outlines
        frame_with_outlines = self.create_frame_with_outlines(
            bleach_frame,
            foci_center,
            nucleus_center,
            self.total_roi_points,
            "bleach_0",
            current_time,
        )
        self.processed_frames.append(
            {"frame": frame_with_outlines, "frame_id": "bleach_0", "time": current_time}
        )

        # Save results
        frame_key = f"bleach_0_{current_time}"
        self.tracked_results[frame_key] = {
            "foci_intensity": foci_intensity,
            "nucleus_intensity": nucleus_intensity,
            "background_intensity": background_intensity,
            "nucleus_center": nucleus_center,
            "foci_center": foci_center,
            "nucleus_roi": self.total_roi_points,
            "foci_radius": (
                self.user_selected_foci_radius
                if self.user_selected_foci_radius is not None
                else self.foci_roi_radius
            ),
        }

        self.bleach_frame_processed = True
        self.current_nucleus_mask = self.reference_nucleus_mask
        print("✅ BLEACH frame processed")

        return self.reference_nucleus_mask, foci_center

    def process_pre_frame(
        self, pre_frame, frame_idx, time_point, reference_frame=None, reference_mask=None
    ):
        """Process PRE frame - sequential tracking from previous PRE frame"""
        if len(pre_frame.shape) == 3:
            frame_gray = (
                pre_frame[0] if pre_frame.shape[0] in [1, 3] else pre_frame.mean(axis=0)
            )
        else:
            frame_gray = pre_frame

        # For the last PRE frame (closest to BLEACH) use BLEACH as reference
        # For other PRE frames use previous PRE frame as reference
        
        if reference_frame is None and reference_mask is None:
            # This is the first processed PRE frame (last in time, closest to BLEACH)
            if self.bleach_frame_gray is not None and self.reference_nucleus_mask is not None:
                tracked_nucleus_mask, nucleus_center = self.track_nucleus_across_frames(
                    self.bleach_frame_gray,  # Reference: BLEACH
                    frame_gray,              # Target: last PRE
                    self.reference_nucleus_mask,  # Mask from BLEACH
                    direction="backward",
                )
            else:
                tracked_nucleus_mask = None
                nucleus_center = None
        else:
            # Subsequent PRE frames (in reverse order): tracking from previous PRE frame
            tracked_nucleus_mask, nucleus_center = self.track_nucleus_across_frames(
                reference_frame,    # Reference: previous PRE frame
                frame_gray,         # Target: current PRE
                reference_mask,     # Mask from previous frame
                direction="backward",
            )

        if tracked_nucleus_mask is not None:
            self.total_roi_points = self.create_nucleus_roi_from_mask(
                tracked_nucleus_mask
            )
            # Update current mask
            self.current_nucleus_mask = tracked_nucleus_mask
        else:
            nucleus_center = None
            self.total_roi_points = None

        # FOCI calculated relative to current nucleus position
        foci_center = None
        if nucleus_center is not None:
            foci_center = self.get_foci_position_from_nucleus(nucleus_center)
        else:
            # If tracking failed, use relative position from bleach
            if self.nucleus_center_bleach is not None:
                nucleus_center = self.nucleus_center_bleach
                foci_center = self.get_foci_position_from_nucleus(nucleus_center)

        # Measure intensities
        if foci_center is not None:
            radius_to_use = (
                self.user_selected_foci_radius
                if self.user_selected_foci_radius is not None
                else self.foci_roi_radius
            )
            foci_intensity = self.measure_intensity_in_roi(
                frame_gray, foci_center, radius_to_use
            )
        else:
            foci_intensity = 0

        nucleus_intensity = (
            self.measure_intensity_in_polygon_roi(frame_gray, self.total_roi_points)
            if self.total_roi_points
            else 0
        )
        background_intensity = (
            self.measure_background_intensity_roi(
                frame_gray, self.background_roi_points
            )
            if self.background_roi_points
            else 0
        )

        print(f"📊 PRE_{frame_idx} (t={time_point:.1f}s): FOCI={foci_intensity:.1f}, NUC={nucleus_intensity:.1f}, BG={background_intensity:.1f}")

        # Create frame with outlines
        frame_with_outlines = self.create_frame_with_outlines(
            pre_frame,
            foci_center,
            nucleus_center,
            self.total_roi_points,
            f"pre_{frame_idx}",
            time_point,
        )
        self.processed_frames.append(
            {
                "frame": frame_with_outlines,
                "frame_id": f"pre_{frame_idx}",
                "time": time_point,
            }
        )

        # Save results
        frame_key = f"pre_{frame_idx}_{time_point}"
        self.tracked_results[frame_key] = {
            "foci_intensity": foci_intensity,
            "nucleus_intensity": nucleus_intensity,
            "background_intensity": background_intensity,
            "nucleus_center": nucleus_center,
            "foci_center": foci_center,
            "nucleus_roi": self.total_roi_points,
            "foci_radius": (
                self.user_selected_foci_radius
                if self.user_selected_foci_radius is not None
                else self.foci_roi_radius
            ),
        }

        return self.current_nucleus_mask, foci_center, frame_gray

    def process_post_frame(
        self, post_frame, frame_idx, time_point, reference_frame=None, reference_mask=None
    ):
        """Process POST frame - sequential tracking from previous POST frame"""
        if len(post_frame.shape) == 3:
            frame_gray = (
                post_frame[0]
                if post_frame.shape[0] in [1, 3]
                else post_frame.mean(axis=0)
            )
        else:
            frame_gray = post_frame

        # For first POST frame use BLEACH as reference
        # For subsequent POST frames use previous POST frame as reference
        if frame_idx == 0:
            # First POST frame: tracking from BLEACH
            if self.bleach_frame_gray is not None and self.reference_nucleus_mask is not None:
                tracked_nucleus_mask, nucleus_center = self.track_nucleus_across_frames(
                    self.bleach_frame_gray,  # Reference: BLEACH
                    frame_gray,              # Target: first POST
                    self.reference_nucleus_mask,  # Mask from BLEACH
                    direction="forward",
                )
            else:
                tracked_nucleus_mask = None
                nucleus_center = None
        else:
            # Subsequent POST frames: tracking from previous POST frame
            tracked_nucleus_mask, nucleus_center = self.track_nucleus_across_frames(
                reference_frame,    # Reference: previous POST frame
                frame_gray,         # Target: current POST
                reference_mask,     # Mask from previous frame
                direction="forward",
            )

        if tracked_nucleus_mask is not None:
            self.total_roi_points = self.create_nucleus_roi_from_mask(
                tracked_nucleus_mask
            )
            # Update current mask
            self.current_nucleus_mask = tracked_nucleus_mask
        else:
            nucleus_center = None
            self.total_roi_points = None

        # FOCI calculated relative to current nucleus position
        foci_center = None
        if nucleus_center is not None:
            foci_center = self.get_foci_position_from_nucleus(nucleus_center)
        else:
            # If tracking failed, use relative position from bleach
            if self.nucleus_center_bleach is not None:
                nucleus_center = self.nucleus_center_bleach
                foci_center = self.get_foci_position_from_nucleus(nucleus_center)

        # Measure intensities
        if foci_center is not None:
            radius_to_use = (
                self.user_selected_foci_radius
                if self.user_selected_foci_radius is not None
                else self.foci_roi_radius
            )
            foci_intensity = self.measure_intensity_in_roi(
                frame_gray, foci_center, radius_to_use
            )
        else:
            foci_intensity = 0

        nucleus_intensity = (
            self.measure_intensity_in_polygon_roi(frame_gray, self.total_roi_points)
            if self.total_roi_points
            else 0
        )
        background_intensity = (
            self.measure_background_intensity_roi(
                frame_gray, self.background_roi_points
            )
            if self.background_roi_points
            else 0
        )

        print(f"📊 POST_{frame_idx} (t={time_point:.1f}s): FOCI={foci_intensity:.1f}, NUC={nucleus_intensity:.1f}, BG={background_intensity:.1f}")

        # Create frame with outlines
        frame_with_outlines = self.create_frame_with_outlines(
            post_frame,
            foci_center,
            nucleus_center,
            self.total_roi_points,
            f"post_{frame_idx}",
            time_point,
        )
        self.processed_frames.append(
            {
                "frame": frame_with_outlines,
                "frame_id": f"post_{frame_idx}",
                "time": time_point,
            }
        )

        # Save results
        frame_key = f"post_{frame_idx}_{time_point}"
        self.tracked_results[frame_key] = {
            "foci_intensity": foci_intensity,
            "nucleus_intensity": nucleus_intensity,
            "background_intensity": background_intensity,
            "nucleus_center": nucleus_center,
            "foci_center": foci_center,
            "nucleus_roi": self.total_roi_points,
            "foci_radius": (
                self.user_selected_foci_radius
                if self.user_selected_foci_radius is not None
                else self.foci_roi_radius
            ),
        }

        return self.current_nucleus_mask, foci_center, frame_gray

    def ask_output_directory(self):
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            messagebox.showinfo(
                "Save", "Select folder to save results"
            )
            output_folder = filedialog.askdirectory(title="Select folder")

            if not output_folder:
                default_folder = os.path.join(os.getcwd(), "FRAP_Analysis_Results")
                os.makedirs(default_folder, exist_ok=True)
                print(f"⚠️ Using default folder: {default_folder}")
                return default_folder

            return output_folder

        except Exception as e:
            print(f"❌ Error: {e}")
            default_folder = os.path.join(os.getcwd(), "FRAP_Analysis_Results")
            os.makedirs(default_folder, exist_ok=True)
            return default_folder

    def save_all_results(self, output_folder):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_folder = os.path.join(output_folder, f"FRAP_Analysis_{timestamp}")
            os.makedirs(results_folder, exist_ok=True)

            print(f"💾 Saving to: {results_folder}")

            if self.tracked_results:
                data = []

                for frame_key, results in self.tracked_results.items():
                    parts = frame_key.split("_")
                    frame_type = parts[0]
                    time_val = float(parts[-1])  # Internal time

                    row = {
                        "Internal_Time_s": time_val,  # For debugging
                        "Foci_Intensity": results["foci_intensity"],
                        "Nucleus_Intensity": results["nucleus_intensity"],
                        "Background_Intensity": results["background_intensity"],
                    }

                    data.append(row)

                df = pd.DataFrame(data)
                
                # Sort by internal time
                df = df.sort_values("Internal_Time_s")
                
                # Convert to absolute time (starting from 0)
                min_time = df["Internal_Time_s"].min()
                df["Time_s"] = df["Internal_Time_s"] - min_time
                
                # Check for time duplicates
                if df["Time_s"].duplicated().any():
                    print("⚠️ Duplicate times detected, adjusting...")
                    epsilon = 0.001
                    for idx, time_val in enumerate(df["Time_s"]):
                        if df["Time_s"].iloc[:idx].eq(time_val).any():
                            df.loc[idx, "Time_s"] = time_val + epsilon
                
                # Rename columns
                df_renamed = df.rename(columns={
                    "Time_s": "Axis [s]",
                    "Foci_Intensity": "ROI1 []",
                    "Nucleus_Intensity": "ROI2 []", 
                    "Background_Intensity": "ROI3 []"
                })

                # CSV with user-defined name: forEasyFRAP_experimentname.csv
                experiment_name = self.experiment_name if self.experiment_name else "experiment"
                csv_filename = f"forEasyFRAP_{experiment_name}.csv"
                csv_path = os.path.join(results_folder, csv_filename)
                
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write("Channel.001\n")
                    f.write("Axis [s],ROI1 [],ROI2 [],ROI3 []\n")
                    for _, row in df_renamed.iterrows():
                        f.write(f"{row['Axis [s]']},{row['ROI1 []']},{row['ROI2 []']},{row['ROI3 []']}\n")
                
                print(f"✅ CSV saved as: {csv_filename}")
                
                # Print time information
                print(f"\n📊 Time information:")
                print(f"   Internal: from {df['Internal_Time_s'].min():.2f} to {df['Internal_Time_s'].max():.2f}s")
                print(f"   Absolute: from {df_renamed['Axis [s]'].min():.2f} to {df_renamed['Axis [s]'].max():.2f}s")
                print(f"   Total time points: {len(df)}")
                print(f"   Frame interval: {self.frame_interval} seconds")

            if self.processed_frames:
                images_folder = os.path.join(results_folder, "Processed_Frames")
                os.makedirs(images_folder, exist_ok=True)

                saved_count = 0
                for frame_data in self.processed_frames:
                    try:
                        frame = frame_data["frame"]
                        frame_id = frame_data["frame_id"]
                        time_point = frame_data["time"]

                        # Simple conversion like in older version
                        safe_frame_id = frame_id.replace(":", "_").replace("/", "_")
                        frame_path = os.path.join(
                            images_folder,
                            f"frame_{safe_frame_id}_time_{time_point:.1f}s.png",
                        )

                        # Minimal checks
                        if frame.dtype != np.uint8:
                            if frame.max() <= 1.0:
                                frame = (frame * 255).astype(np.uint8)
                            else:
                                frame = frame.astype(np.uint8)

                        pil_image = Image.fromarray(frame)
                        pil_image.save(frame_path, "PNG")
                        saved_count += 1

                    except Exception as e:
                        print(f"❌ Error saving frame: {e}")
                        continue

                print(f"✅ Images: {saved_count} frames saved")

            params_path = os.path.join(results_folder, "Parameters.txt")
            with open(params_path, "w", encoding="utf-8") as f:
                f.write(f"Frame_Interval: {self.frame_interval} seconds\n")
                f.write(f"Experiment_Name: {self.experiment_name}\n")
                f.write(f"Foci_ROI_Radius: {self.foci_roi_radius}\n")
                f.write(
                    f"User_Selected_Foci_Radius: {self.user_selected_foci_radius}\n"
                )
                f.write(f"Foci_Center_Bleach: {self.foci_roi_center_bleach}\n")
                f.write(f"Nucleus_Center_Bleach: {self.nucleus_center_bleach}\n")
                f.write(f"Relative_Foci_Position: {self.relative_foci_position}\n")

            print(f"✅ All results saved")
            return results_folder

        except Exception as e:
            print(f"❌ Error saving: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_frap_analysis_2d(self):
        print("🚀 STARTING FRAP ANALYSIS...")

        # Ask for experiment name first
        print("\n--- 📝 REQUEST EXPERIMENT NAME ---")
        self.experiment_name = self.ask_experiment_name()

        image_files = self.load_image_files()
        if not image_files:
            print("❌ File loading cancelled")
            return

        try:
            print("📁 Loading images...")
            pre_bleach_image = io.imread(image_files["PRE_BLEACH"])
            bleach_image = io.imread(image_files["BLEACH"])
            post_bleach_image = io.imread(image_files["POST_BLEACH"])

            print(f"✅ Images loaded:")
            print(f"   PRE: {pre_bleach_image.shape}")
            print(f"   BLEACH: {bleach_image.shape}")
            print(f"   POST: {post_bleach_image.shape}")

        except Exception as e:
            print(f"❌ Error loading images: {e}")
            return

        # Ask for frame interval
        print("\n--- ⏰ REQUEST FRAME INTERVAL ---")
        frame_interval = self.ask_frame_interval()
        if frame_interval is None:
            print("❌ User cancelled interval input")
            return
        
        self.frame_interval = frame_interval
        print(f"✅ Interval set: {frame_interval} seconds")

        if not self.select_all_rois_on_bleach(bleach_image):
            print("❌ ROI selection on BLEACH cancelled")
            return

        print("✅ FOCI and NUCLEUS ROIs SELECTED ON BLEACH IMAGE")

        print("\n--- 🔵 ENHANCED BACKGROUND ROI SELECTION WITH SLIDER ---")
        if not self.select_background_roi_with_slider(
            pre_bleach_image, bleach_image, post_bleach_image
        ):
            print("❌ BACKGROUND ROI selection cancelled")
            return

        print("✅ BACKGROUND ROI SELECTED WITH ALL FRAMES VIEWING CAPABILITY")

        print("⏰ Preparing time points (simplified system)...")
        
        # Determine number of frames
        if len(pre_bleach_image.shape) == 3:
            pre_frames = pre_bleach_image.shape[0]
        else:
            pre_frames = 1
            
        if len(post_bleach_image.shape) == 3:
            post_frames = post_bleach_image.shape[0]
        else:
            post_frames = 1
        
        print(f"📊 Detected frames: PRE={pre_frames}, POST={post_frames}")
        
        # Create time points with new system:
        # Bleach = 0, Pre = negative, Post = positive
        bleach_time = 0
        
        # PRE times (negative)
        pre_times = []
        for i in range(pre_frames):
            # Last PRE frame should be right before bleach
            time_val = -(pre_frames - i) * self.frame_interval
            pre_times.append(time_val)
        
        # POST times (positive)
        post_times = []
        for i in range(post_frames):
            time_val = (i + 1) * self.frame_interval
            post_times.append(time_val)
        
        print(f"📊 Time range: from {pre_times[0] if pre_times else 0:.1f} to {post_times[-1] if post_times else 0:.1f}s")

        # Save first PRE frame for possible alternative tracking
        if len(pre_bleach_image.shape) == 3:
            self.first_pre_frame_gray = pre_bleach_image[0]
            if len(self.first_pre_frame_gray.shape) == 3:
                self.first_pre_frame_gray = self.first_pre_frame_gray.mean(axis=2)
        else:
            self.first_pre_frame_gray = pre_bleach_image

        # 1. PROCESS BLEACH FRAME (time = 0)
        print(f"\n📊 PROCESSING BLEACH FRAME (t={bleach_time}s)...")
        if len(bleach_image.shape) == 3:
            bleach_frame = bleach_image[0]
        else:
            bleach_frame = bleach_image

        current_nucleus_mask, _ = self.process_bleach_frame(bleach_frame, bleach_time)

        # 2. PROCESS PRE FRAMES (negative times)
        if pre_frames > 0:
            print(f"\n📊 PROCESSING PRE FRAMES ({pre_frames} frames)...")
            if len(pre_bleach_image.shape) == 3:
                # Process in reverse order: from last to first
                previous_pre_frame_gray = None
                previous_nucleus_mask = None
                
                for i in range(pre_bleach_image.shape[0] - 1, -1, -1):
                    frame = pre_bleach_image[i]
                    time = pre_times[i] if i < len(pre_times) else -(pre_frames - i) * self.frame_interval

                    current_nucleus_mask, foci_center, current_frame_gray = self.process_pre_frame(
                        frame, i, time, previous_pre_frame_gray, previous_nucleus_mask
                    )
                    
                    # Update previous frame for next iteration
                    previous_pre_frame_gray = current_frame_gray
                    previous_nucleus_mask = current_nucleus_mask
            else:
                current_nucleus_mask, foci_center, _ = self.process_pre_frame(
                    pre_bleach_image, 0, pre_times[0], None, None
                )

        # 3. RESTORE nucleus mask for POST (original from BLEACH)
        current_nucleus_mask = self.reference_nucleus_mask
        print("🔄 Restored BLEACH mask for POST tracking")

        # 4. PROCESS POST FRAMES (positive times)
        if post_frames > 0:
            print(f"\n📊 PROCESSING POST FRAMES ({post_frames} frames)...")
            if len(post_bleach_image.shape) == 3:
                # Save previous frame for tracking
                previous_post_frame_gray = None
                previous_nucleus_mask = self.reference_nucleus_mask
                
                for i in range(post_bleach_image.shape[0]):
                    frame = post_bleach_image[i]
                    time = post_times[i] if i < len(post_times) else (i + 1) * self.frame_interval

                    current_nucleus_mask, foci_center, current_frame_gray = self.process_post_frame(
                        frame, i, time, previous_post_frame_gray, previous_nucleus_mask
                    )
                    
                    # Update previous frame for next iteration
                    previous_post_frame_gray = current_frame_gray
                    previous_nucleus_mask = current_nucleus_mask
            else:
                current_nucleus_mask, foci_center, _ = self.process_post_frame(
                    post_bleach_image, 0, post_times[0], None, self.reference_nucleus_mask
                )

        print("\n📁 SELECTING OUTPUT FOLDER...")
        output_folder = self.ask_output_directory()

        print("\n💾 SAVING ALL RESULTS...")
        final_results_folder = self.save_all_results(output_folder)

        print("\n📊 Visualizing results...")
        self.visualize_results()

        print("✅ FRAP ANALYSIS COMPLETED!")

        if final_results_folder:
            messagebox.showinfo(
                "Analysis Complete",
                f"FRAP analysis successfully completed!\n\n"
                f"📊 Frames processed: {len(self.tracked_results)}\n"
                f"🖼️ Images saved: {len(self.processed_frames)}\n"
                f"⏱️ Frame interval: {self.frame_interval} sec\n"
                f"📝 Experiment: {self.experiment_name}\n"
                f"💾 Results saved in:\n{final_results_folder}",
            )

    def visualize_results(self):
        try:
            if not self.tracked_results:
                print("❌ No data for visualization")
                return

            times = []
            foci_intensities = []
            nucleus_intensities = []
            background_intensities = []
            nucleus_x = []
            nucleus_y = []

            for frame_key, results in self.tracked_results.items():
                parts = frame_key.split("_")
                if len(parts) >= 2:
                    time_str = parts[-1]
                    times.append(float(time_str))
                    foci_intensities.append(results["foci_intensity"])
                    nucleus_intensities.append(results["nucleus_intensity"])
                    background_intensities.append(results["background_intensity"])
                    if results["nucleus_center"] is not None:
                        nucleus_x.append(results["nucleus_center"][0])
                        nucleus_y.append(results["nucleus_center"][1])
                    else:
                        nucleus_x.append(0)
                        nucleus_y.append(0)

            sorted_data = sorted(
                zip(
                    times,
                    foci_intensities,
                    nucleus_intensities,
                    background_intensities,
                    nucleus_x,
                    nucleus_y,
                )
            )
            (
                times_sorted,
                foci_sorted,
                nucleus_sorted,
                background_sorted,
                x_sorted,
                y_sorted,
            ) = zip(*sorted_data)

            fig, axes = plt.subplots(2, 1, figsize=(12, 10))

            # Intensity plot
            ax1 = axes[0]
            ax1.plot(
                times_sorted,
                foci_sorted,
                "ro-",
                linewidth=2,
                markersize=6,
                label="FOCI",
            )
            ax1.plot(
                times_sorted,
                nucleus_sorted,
                "go-",
                linewidth=2,
                markersize=6,
                label="NUCLEUS",
            )
            ax1.plot(
                times_sorted,
                background_sorted,
                "bo-",
                linewidth=2,
                markersize=6,
                label="BACKGROUND",
            )
            ax1.axvline(
                x=0, color="red", linestyle="--", alpha=0.7, label="Bleach moment"
            )
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Intensity")
            ax1.set_title("FRAP Analysis - Intensities (internal times)")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Nucleus movement plot
            ax2 = axes[1]
            ax2.plot(
                times_sorted,
                x_sorted,
                "bo-",
                linewidth=2,
                markersize=4,
                label="X coordinate",
            )
            ax2.plot(
                times_sorted,
                y_sorted,
                "ro-",
                linewidth=2,
                markersize=4,
                label="Y coordinate",
            )
            ax2.axvline(
                x=0, color="red", linestyle="--", alpha=0.7, label="Bleach moment"
            )
            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Coordinate (pixels)")
            ax2.set_title("Nucleus geometric center movement")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.show()

            print("✅ Visualization complete")

            # Print movement statistics
            print(f"\n📊 Nucleus movement statistics (geometric center):")
            print(
                f"   X: min={min(x_sorted):.1f}, max={max(x_sorted):.1f}, disp={max(x_sorted)-min(x_sorted):.1f}px"
            )
            print(
                f"   Y: min={min(y_sorted):.1f}, max={max(y_sorted):.1f}, disp={max(y_sorted)-min(y_sorted):.1f}px"
            )

        except Exception as e:
            print(f"❌ Error in visualization: {e}")


def main():
    print("=" * 60)
    print("         🧪 FRAP-Tracker BASIC by IGB 🎯")
    print("      Fixed FOCI relative to nucleus - V2024.12.06")
    print("=" * 60)

    try:
        analyzer = FRAPTrackerBasic()
        analyzer.run_frap_analysis_2d()
        print("\n🎉 ANALYSIS COMPLETE!")

    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        pass


if __name__ == "__main__":
    main()