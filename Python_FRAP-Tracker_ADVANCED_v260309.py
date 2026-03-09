# =============================================================================
#                     🧪 FRAP-Tracker ADCANCED by IGB 🎯
#      Semi-automatic segmentation with nucleus tracking (2D VERSION)
# =============================================================================

# ==================== IMPORTS AND SETTINGS ====================
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import numpy as np
import pandas as pd
from skimage import io, measure, morphology, feature, filters, segmentation, draw
from scipy import ndimage
from scipy.spatial import distance
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Polygon, Circle, Rectangle
import os
import datetime
from PIL import Image, ImageDraw, ImageFont

# ==================== BLOCK 0: ANALYZER CLASS ====================
class FRAPTrackerIGB:
    def __init__(self):
        """Initialize FRAP-Tracker IGB for 2D analysis"""
        self.root = tk.Tk()
        self.root.withdraw()
        self.selected_bleach_center = None
        self.total_roi_points = None
        self.background_roi_points = None
        self.reference_nucleus_mask = None
        self.reference_center = None
        self.reference_area = None
        self.background_frame_indices = []
        self.foci_frame_indices = []
        self.tracked_results = {}
        self.foci_roi_radius = None
        self.foci_roi_center_first_pre = None
        self.csv_base_name = "forEasyFRAP"
        self.bleach_foci_center = None  # Сохраняем выбранный центр на BLEACH

    def load_image_files(self):
        """Load all required 2D images through dialogs"""
        print("📁 Loading image files...")
        
        image_files = {}
        
        file_types = [
            ('PRE_BLEACH', 'PRE BLEACH images', 'Select PRE BLEACH file (TIFF)'),
            ('BLEACH', 'BLEACH images', 'Select BLEACH file (TIFF)'),
            ('POST_BLEACH', 'POST BLEACH images', 'Select POST BLEACH file (TIFF)'),
            ('MASK_BLEACH', 'MASK images', 'Select MASK file (TIFF)')
        ]
        
        for file_key, file_desc, dialog_title in file_types:
            while True:
                messagebox.showinfo(f"Select {file_desc}", dialog_title)
                file_path = filedialog.askopenfilename(
                    title=dialog_title,
                    filetypes=[("TIFF files", "*.tif;*.tiff"), ("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")]
                )
                
                if not file_path:
                    if messagebox.askretrycancel("File not selected", 
                                               f"{file_desc} file not selected. Try again?"):
                        continue
                    else:
                        return None
                else:
                    if os.path.exists(file_path):
                        image_files[file_key] = file_path
                        print(f"✅ {file_desc}: {os.path.basename(file_path)}")
                        break
                    else:
                        messagebox.showerror("Error", f"File does not exist: {file_path}")
        
        return image_files

    def create_and_validate_mask(self, mask_image):
        """Automatic 2D mask binarization with validation"""
        print("🔄 Automatic mask binarization...")
        
        if len(mask_image.shape) == 3:
            mask_image = mask_image.mean(axis=2)
        
        threshold_value = filters.threshold_otsu(mask_image)
        binary_mask = mask_image > threshold_value
        
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=50)
        binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=100)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.imshow(mask_image, cmap='gray')
        ax1.set_title('Original mask')
        ax1.axis('off')
        
        ax2.imshow(binary_mask, cmap='gray')
        ax2.set_title('Binarized mask')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        result = messagebox.askyesno(
            "Mask validation", 
            "Automatic binarization completed.\n\n"
            "✅ LEFT PANEL: Original image\n"
            "✅ RIGHT PANEL: Binarized mask\n\n"
            "Is binarization correct?",
            detail="Click 'YES' to continue or 'NO' for manual selection"
        )
        
        if result:
            print("✅ Binarization accepted by user")
            return binary_mask
        else:
            print("🔄 Manual mask selection...")
            return self.manual_mask_selection(mask_image)

    def manual_mask_selection(self, mask_image):
        """Manual mask selection (placeholder)"""
        messagebox.showinfo("Manual selection", "Manual mask selection not yet implemented")
        return None

    def detect_roi_from_binary_mask(self, binary_mask):
        """Detect ROI from binary 2D mask"""
        print("🎯 Detecting ROI from binary mask...")
        
        contours = measure.find_contours(binary_mask, 0.5)
        
        if not contours:
            messagebox.showerror("Error", "No contours found in binary mask!")
            return None, None
        
        largest_contour = max(contours, key=len)
        
        center_y, center_x = np.mean(largest_contour, axis=0)
        
        distances = [distance.euclidean([center_x, center_y], [x, y]) for y, x in largest_contour]
        radius = np.mean(distances)
        
        print(f"✅ ROI detected: center ({center_x:.1f}, {center_y:.1f}), radius {radius:.1f} pixels")
        
        return (center_x, center_y), radius

    def select_foci_roi_on_first_pre_bleach(self, pre_bleach_image, bleach_image):
        """Select FOCI ROI on FIRST PRE BLEACH frame with comparison to BLEACH"""
        print("🎯 MANUAL FOCI ROI SELECTION ON FIRST PRE BLEACH FRAME...")
        
        class FociROISelectionState:
            def __init__(self, pre_image, bleach_image):
                self.pre_image = pre_image
                self.bleach_image = bleach_image
                self.center = None
                self.radius = None
                self.finished = False
        
        # Take FIRST PRE BLEACH frame
        if len(pre_bleach_image.shape) == 3:
            first_pre_frame = pre_bleach_image[0]
        else:
            first_pre_frame = pre_bleach_image
            
        state = FociROISelectionState(first_pre_frame, bleach_image)
        
        # Create figure with two panels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        plt.subplots_adjust(bottom=0.2)
        
        # Prepare images for display
        if len(state.pre_image.shape) == 3:
            pre_display = state.pre_image[0] if state.pre_image.shape[0] in [1, 3] else state.pre_image.mean(axis=0)
        else:
            pre_display = state.pre_image
            
        if len(state.bleach_image.shape) == 3:
            bleach_display = state.bleach_image[0] if state.bleach_image.shape[0] in [1, 3] else state.bleach_image.mean(axis=0)
        else:
            bleach_display = state.bleach_image
        
        pre_display = self.optimize_contrast_for_display(pre_display)
        bleach_display = self.optimize_contrast_for_display(bleach_display)
        
        # Function to update display
        def update_display():
            ax1.clear()
            ax2.clear()
            
            ax1.imshow(pre_display, cmap='gray')
            ax2.imshow(bleach_display, cmap='gray')
            
            # Draw selected ROI on PRE image
            if state.center and state.radius:
                circle_pre = Circle(state.center, state.radius, fill=False, color='red', linewidth=2)
                ax1.add_patch(circle_pre)
                ax1.plot(state.center[0], state.center[1], 'r+', markersize=15, markeredgewidth=2)
                ax1.text(state.center[0] + state.radius + 5, state.center[1], f'R: {state.radius:.1f}px', 
                        color='red', fontsize=10, fontweight='bold')
                
                # Also draw on BLEACH for comparison
                circle_bleach = Circle(state.center, state.radius, fill=False, color='red', linewidth=2, alpha=0.7)
                ax2.add_patch(circle_bleach)
                ax2.plot(state.center[0], state.center[1], 'r+', markersize=15, markeredgewidth=2, alpha=0.7)
            
            ax1.set_title('PRE BLEACH (FIRST frame)\nSelect FOCI ROI: click for center', 
                         fontsize=12, fontweight='bold', color='red')
            ax2.set_title('BLEACH (for comparison)\nROI will be shown for reference', 
                         fontsize=12, fontweight='bold', color='blue')
            
            ax1.axis('off')
            ax2.axis('off')
            
            plt.draw()
        
        # Mouse click handler (only on left panel)
        def on_click(event):
            if event.inaxes == ax1 and event.button == 1:  # Left click only on PRE image
                state.center = (event.xdata, event.ydata)
                print(f"✅ FOCI center selected on first PRE frame: ({event.xdata:.1f}, {event.ydata:.1f})")
                update_display()
        
        # Slider for radius
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        radius_slider = Slider(ax_slider, 'FOCI Radius (pixels)', 1, 20, valinit=5, valfmt='%.0f')
        
        # Buttons
        ax_finish = plt.axes([0.3, 0.05, 0.2, 0.04])
        ax_cancel = plt.axes([0.55, 0.05, 0.15, 0.04])
        
        btn_finish = Button(ax_finish, '✅ Complete selection', color='lightgreen')
        btn_cancel = Button(ax_cancel, '❌ Cancel', color='lightcoral')
        
        def on_slider_change(val):
            state.radius = val
            update_display()
        
        def finish_selection(event):
            if not state.center or not state.radius:
                messagebox.showwarning("Incomplete", "Select FOCI center and radius!")
                return
            state.finished = True
            plt.close()
            print(f"✅ FOCI ROI selected on first PRE frame: center {state.center}, radius {state.radius:.1f} pixels")
        
        def cancel_selection(event):
            state.finished = False
            plt.close()
            print("❌ FOCI ROI selection cancelled")
        
        # Initialization
        state.radius = radius_slider.val
        
        # Connect handlers
        radius_slider.on_changed(on_slider_change)
        btn_finish.on_clicked(finish_selection)
        btn_cancel.on_clicked(cancel_selection)
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        # Initial display
        update_display()
        
        print("\n🎯 FOCI ROI SELECTION INSTRUCTIONS:")
        print("1. LEFT PANEL: PRE BLEACH (FIRST frame) - click to select FOCI center")
        print("2. Use slider to select ROI radius")
        print("3. RIGHT PANEL: BLEACH - for comparison only")
        print("4. Click '✅ Complete selection' when done")
        
        plt.show()
        
        if state.finished:
            self.foci_roi_center_first_pre = state.center
            self.foci_roi_radius = state.radius
            return True
        else:
            return False

    def manual_nucleus_selection_2d(self, bleach_image):
        """MANUAL NUCLEUS SELECTION by user on BLEACH image"""
        print("🎯 MANUAL NUCLEUS SELECTION...")
        
        class NucleusSelectionState:
            def __init__(self):
                self.points = []
                self.finished = False
        
        state = NucleusSelectionState()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.15)
        
        if len(bleach_image.shape) == 3:
            display_image = bleach_image[0] if bleach_image.shape[0] in [1, 3] else bleach_image.mean(axis=0)
        else:
            display_image = bleach_image
        
        display_image = self.optimize_contrast_for_display(display_image)
        
        def update_display():
            ax.clear()
            ax.imshow(display_image, cmap='gray')
            
            if state.points:
                ax.plot([p[0] for p in state.points], [p[1] for p in state.points], 'ro', markersize=6, markerfacecolor='red')
                
                if len(state.points) > 1:
                    points_array = np.array(state.points)
                    ax.plot(points_array[:, 0], points_array[:, 1], 'r-', linewidth=2, alpha=0.7)
                
                if len(state.points) > 2:
                    closed_points = state.points + [state.points[0]]
                    closed_array = np.array(closed_points)
                    ax.plot(closed_array[:, 0], closed_array[:, 1], 'r--', linewidth=1, alpha=0.5)
            
            ax.set_title('MANUAL NUCLEUS SELECTION: Outline nucleus with left clicks\nComplete selection with button below', 
                        fontsize=14, fontweight='bold', color='red')
            ax.text(0.02, 0.98, f'Points: {len(state.points)}', 
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
            
            instructions = [
                "📝 INSTRUCTIONS:",
                "1. Left-click on nucleus contour",
                "2. Outline entire nucleus perimeter", 
                "3. Place points evenly",
                "4. Click '✅ COMPLETE SELECTION' when done"
            ]
            
            for i, instruction in enumerate(instructions):
                ax.text(0.02, 0.90 - i*0.04, instruction, 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.axis('off')
            plt.draw()
        
        def on_click(event):
            if event.inaxes == ax and event.button == 1:
                state.points.append([event.xdata, event.ydata])
                print(f"✅ Nucleus contour point added: ({event.xdata:.1f}, {event.ydata:.1f})")
                update_display()
        
        ax_remove_point = plt.axes([0.3, 0.05, 0.2, 0.05])
        ax_finish_selection = plt.axes([0.6, 0.05, 0.3, 0.05])
        
        btn_remove_point = Button(ax_remove_point, '❌ Remove point', color='lightcoral')
        btn_finish_selection = Button(ax_finish_selection, '✅ COMPLETE SELECTION', color='lightgreen')
        
        def remove_point(event):
            if state.points:
                removed_point = state.points.pop()
                print(f"❌ Contour point removed: ({removed_point[0]:.1f}, {removed_point[1]:.1f})")
                update_display()
        
        def finish_selection(event):
            if len(state.points) < 5:
                messagebox.showwarning("Too few points", "Need at least 5 points for accurate nucleus contour!")
                return
            
            if state.points[0] != state.points[-1]:
                state.points.append(state.points[0])
            
            state.finished = True
            plt.close()
            print(f"✅ Manual nucleus selection completed: {len(state.points)} points")
        
        btn_remove_point.on_clicked(remove_point)
        btn_finish_selection.on_clicked(finish_selection)
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        update_display()
        
        print("\n🎯 MANUAL NUCLEUS SELECTION:")
        print("1. Outline nucleus contour with left clicks")
        print("2. Place points around nucleus perimeter")
        print("3. Click '✅ COMPLETE SELECTION' when done")
        
        plt.show()
        
        if state.finished:
            from skimage.draw import polygon
            mask = np.zeros(display_image.shape, dtype=bool)
            rr, cc = polygon([p[1] for p in state.points], [p[0] for p in state.points], display_image.shape)
            mask[rr, cc] = True
            
            self.reference_nucleus_mask = mask
            self.reference_center = np.array([np.mean([p[0] for p in state.points]), 
                                            np.mean([p[1] for p in state.points])])
            self.reference_area = np.sum(mask)
            
            print(f"✅ Reference nucleus saved: area {self.reference_area} pixels")
            
            return True
        else:
            return False

    def track_nucleus_across_frames(self, reference_frame, target_frame, reference_mask):
        """Track nucleus between frames while preserving area"""
        try:
            if len(reference_frame.shape) == 3:
                reference_frame_gray = reference_frame.mean(axis=2)
            else:
                reference_frame_gray = reference_frame
                
            if len(target_frame.shape) == 3:
                target_frame_gray = target_frame.mean(axis=2)
            else:
                target_frame_gray = target_frame
            
            reference_center = np.array(ndimage.center_of_mass(reference_mask))
            
            try:
                correlation = feature.match_template(target_frame_gray, reference_frame_gray, pad_input=True)
                max_pos = np.unravel_index(np.argmax(correlation), correlation.shape)
                displacement = np.array(max_pos) - np.array(reference_frame_gray.shape) / 2
                new_center = reference_center + displacement
                
                from scipy.ndimage import shift
                tracked_mask = shift(reference_mask.astype(float), displacement, order=0) > 0.5
                
                current_area = np.sum(tracked_mask)
                
                if self.reference_area and abs(current_area - self.reference_area) > self.reference_area * 0.1:
                    if current_area < self.reference_area:
                        from skimage.morphology import dilation, disk
                        tracked_mask = dilation(tracked_mask, disk(1))
                    else:
                        from skimage.morphology import erosion, disk
                        tracked_mask = erosion(tracked_mask, disk(1))
                    
                    for _ in range(5):
                        current_area = np.sum(tracked_mask)
                        if abs(current_area - self.reference_area) < self.reference_area * 0.05:
                            break
                        
                        if current_area < self.reference_area:
                            tracked_mask = dilation(tracked_mask, disk(1))
                        else:
                            tracked_mask = erosion(tracked_mask, disk(1))
                
                return tracked_mask, new_center
                
            except Exception as e:
                return reference_mask, reference_center
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return reference_mask, np.array(ndimage.center_of_mass(reference_mask))

    def create_nucleus_roi_from_mask(self, nucleus_mask):
        """Create ROI from nucleus mask"""
        if nucleus_mask is None:
            return None
            
        contours = measure.find_contours(nucleus_mask, 0.5)
        
        if not contours:
            return None
        
        main_contour = max(contours, key=len)
        simplified_contour = main_contour[::3]
        roi_points = simplified_contour[:, [1, 0]].tolist()
        
        if roi_points[0] != roi_points[-1]:
            roi_points.append(roi_points[0])
        
        return roi_points

    def select_background_roi_with_slider(self, image_stack):
        """Select background ROI with frame slider"""
        print("🔄 Selecting BACKGROUND ROI with slider...")
        
        class BackgroundROIState:
            def __init__(self, stack):
                self.stack = stack
                self.current_frame_idx = 0
                self.points = []
                self.finished = False
        
        state = BackgroundROIState(image_stack)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.25)
        
        def show_current_frame():
            ax.clear()
            current_frame = state.stack[state.current_frame_idx]
            
            if len(current_frame.shape) == 3:
                display_frame = current_frame[0] if current_frame.shape[0] in [1, 3] else current_frame.mean(axis=0)
            else:
                display_frame = current_frame
            
            display_frame = self.optimize_contrast_for_display(display_frame)
            ax.imshow(display_frame, cmap='gray')
            
            if state.points:
                ax.plot([p[0] for p in state.points], [p[1] for p in state.points], 'bo', markersize=6)
                
                if len(state.points) > 1:
                    points_array = np.array(state.points)
                    ax.plot(points_array[:, 0], points_array[:, 1], 'b-', linewidth=2)
                
                if len(state.points) > 2:
                    closed_points = state.points + [state.points[0]]
                    closed_array = np.array(closed_points)
                    ax.plot(closed_array[:, 0], closed_array[:, 1], 'b--', linewidth=1, alpha=0.7)
            
            ax.set_title(f'BACKGROUND ROI - Frame {state.current_frame_idx + 1}/{len(state.stack)}\nSelect area WITHOUT cells')
            ax.text(0.02, 0.98, f'Points: {len(state.points)}', 
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='blue', alpha=0.8))
            
            plt.draw()
        
        ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
        slider = Slider(ax_slider, 'Frame', 1, len(state.stack), valinit=1, valfmt='%d')
        
        ax_add_point = plt.axes([0.1, 0.05, 0.15, 0.05])
        ax_remove_point = plt.axes([0.3, 0.05, 0.15, 0.05])
        ax_finish_roi = plt.axes([0.5, 0.05, 0.2, 0.05])
        ax_cancel = plt.axes([0.75, 0.05, 0.15, 0.05])
        
        btn_add_point = Button(ax_add_point, '➕ Add point')
        btn_remove_point = Button(ax_remove_point, '❌ Remove point', color='lightcoral')
        btn_finish_roi = Button(ax_finish_roi, '✅ Complete selection', color='lightgreen')
        btn_cancel = Button(ax_cancel, '❌ Cancel', color='lightcoral')
        
        def on_slider_change(val):
            state.current_frame_idx = int(val) - 1
            show_current_frame()
        
        def on_click(event):
            if event.inaxes == ax and event.button == 1:
                state.points.append([event.xdata, event.ydata])
                print(f"✅ BACKGROUND point added: ({event.xdata:.1f}, {event.ydata:.1f})")
                show_current_frame()
        
        def remove_point(event):
            if state.points:
                removed_point = state.points.pop()
                print(f"❌ BACKGROUND point removed: ({removed_point[0]:.1f}, {removed_point[1]:.1f})")
                show_current_frame()
        
        def finish_roi(event):
            if len(state.points) < 3:
                messagebox.showwarning("Too few points", "Need at least 3 points for ROI!")
                return
            
            if state.points[0] != state.points[-1]:
                state.points.append(state.points[0])
            
            state.finished = True
            plt.close()
            print(f"✅ BACKGROUND ROI completed: {len(state.points)} points")
            self.background_frame_indices.append(state.current_frame_idx)
        
        def cancel_selection(event):
            state.points = []
            state.finished = False
            plt.close()
            print("❌ BACKGROUND ROI selection cancelled")
        
        slider.on_changed(on_slider_change)
        btn_add_point.on_clicked(lambda x: None)
        btn_remove_point.on_clicked(remove_point)
        btn_finish_roi.on_clicked(finish_roi)
        btn_cancel.on_clicked(cancel_selection)
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        show_current_frame()
        
        print("\n🖱️ BACKGROUND ROI SELECTION:")
        print("1. Use slider to navigate frames")
        print("2. Left-click to add points")
        print("3. Select area WITHOUT cells for background measurement")
        print("4. Click '✅ Complete selection' when done")
        
        plt.show()
        
        if state.finished:
            self.background_roi_points = state.points
            return True
        else:
            self.background_roi_points = None
            return False

    def select_foci_position_with_slider(self, image_stack):
        """Select FOCI position on BLEACH image with slider"""
        print("🔄 Selecting FOCI position on BLEACH with slider...")
        
        class FociSelectionState:
            def __init__(self, stack):
                self.stack = stack
                self.current_frame_idx = 0
                self.selected_center = None
                self.finished = False
        
        state = FociSelectionState(image_stack)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.2)
        
        def show_current_frame():
            ax.clear()
            current_frame = state.stack[state.current_frame_idx]
            
            if len(current_frame.shape) == 3:
                display_frame = current_frame[0] if current_frame.shape[0] in [1, 3] else current_frame.mean(axis=0)
            else:
                display_frame = current_frame
            
            display_frame = self.optimize_contrast_for_display(display_frame)
            ax.imshow(display_frame, cmap='gray')
            
            if state.selected_center:
                center_x, center_y = state.selected_center
                ax.plot(center_x, center_y, 'r+', markersize=20, markeredgewidth=3)
                circle = Circle(state.selected_center, self.foci_roi_radius, fill=False, color='red', linewidth=2)
                ax.add_patch(circle)
                ax.text(center_x + 10, center_y + 10, 'FOCI', color='red', fontsize=12, fontweight='bold')
                ax.text(center_x + 10, center_y - 10, f'R: {self.foci_roi_radius:.1f}px', 
                       color='red', fontsize=10, fontweight='bold')
            
            ax.set_title(f'FOCI POSITION SELECTION ON BLEACH - Frame {state.current_frame_idx + 1}/{len(state.stack)}\nLeft-click to select position')
            if state.selected_center:
                ax.text(0.02, 0.98, f'Position: {state.selected_center}', 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
            
            plt.draw()
        
        ax_slider = plt.axes([0.2, 0.12, 0.6, 0.03])
        slider = Slider(ax_slider, 'Frame', 1, len(state.stack), valinit=1, valfmt='%d')
        
        ax_finish_selection = plt.axes([0.3, 0.05, 0.2, 0.05])
        ax_cancel = plt.axes([0.55, 0.05, 0.15, 0.05])
        
        btn_finish_selection = Button(ax_finish_selection, '✅ Complete selection', color='lightgreen')
        btn_cancel = Button(ax_cancel, '❌ Cancel', color='lightcoral')
        
        def on_slider_change(val):
            state.current_frame_idx = int(val) - 1
            show_current_frame()
        
        def on_click(event):
            if event.inaxes == ax and event.button == 1:
                state.selected_center = (event.xdata, event.ydata)
                print(f"✅ FOCI position selected on BLEACH: ({event.xdata:.1f}, {event.ydata:.1f}) on frame {state.current_frame_idx + 1}")
                show_current_frame()
        
        def finish_selection(event):
            if not state.selected_center:
                messagebox.showwarning("Position not selected", "Select FOCI position with mouse click!")
                return
            
            state.finished = True
            plt.close()
            print(f"✅ FOCI selection on BLEACH completed: position {state.selected_center}")
            self.foci_frame_indices.append(state.current_frame_idx)
            self.bleach_foci_center = state.selected_center  # Сохраняем выбранный центр
            
        def cancel_selection(event):
            state.selected_center = None
            state.finished = False
            plt.close()
            print("❌ FOCI selection on BLEACH cancelled")
        
        slider.on_changed(on_slider_change)
        btn_finish_selection.on_clicked(finish_selection)
        btn_cancel.on_clicked(cancel_selection)
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        show_current_frame()
        
        print("\n🖱️ FOCI POSITION SELECTION ON BLEACH:")
        print("1. Use slider to navigate frames")
        print("2. Left-click to select FOCI position")
        print("3. Click '✅ Complete selection' when done")
        
        plt.show()
        
        if state.finished:
            self.selected_bleach_center = state.selected_center
            return True
        else:
            self.selected_bleach_center = None
            return False

    def optimize_contrast_for_display(self, image):
        """Optimize contrast for 2D image display"""
        p2, p98 = np.percentile(image, (2, 98))
        if p98 - p2 > 0:
            normalized = np.clip((image - p2) / (p98 - p2), 0, 1)
        else:
            normalized = image / image.max() if image.max() > 0 else image
        return normalized

    def measure_intensity_in_roi(self, image, center, radius):
        """Measure intensity in circular ROI"""
        center_x, center_y = center
        y_indices, x_indices = np.ogrid[:image.shape[0], :image.shape[1]]
        mask = ((x_indices - center_x)**2 + (y_indices - center_y)**2) <= radius**2
        
        if np.sum(mask) == 0:
            return 0
            
        return np.mean(image[mask])

    def measure_background_intensity_roi(self, image, background_points):
        """Measure background intensity in polygonal ROI"""
        if not background_points or len(background_points) < 3:
            return 0
            
        from skimage.draw import polygon
        roi_mask = np.zeros(image.shape[:2], dtype=bool)
        rr, cc = polygon([p[1] for p in background_points], [p[0] for p in background_points], image.shape[:2])
        roi_mask[rr, cc] = True
        
        if len(image.shape) == 3:
            image = image.mean(axis=2)
        
        intensity = np.mean(image[roi_mask])
        return intensity

    def find_best_foci_position(self, frame, search_center, search_radius, foci_radius):
        """Find best FOCI position in search area by maximum average intensity in ROI"""
        center_x, center_y = search_center
        
        search_margin = int(search_radius)
        x_min = max(0, int(center_x - search_margin))
        x_max = min(frame.shape[1], int(center_x + search_margin))
        y_min = max(0, int(center_y - search_margin))
        y_max = min(frame.shape[0], int(center_y + search_margin))
        
        best_intensity = -1
        best_position = search_center
        
        for x in range(x_min, x_max, 2):
            for y in range(y_min, y_max, 2):
                if ((x - center_x)**2 + (y - center_y)**2) <= search_radius**2:
                    intensity = self.measure_intensity_in_roi(frame, (x, y), foci_radius)
                    if intensity > best_intensity:
                        best_intensity = intensity
                        best_position = (x, y)
        
        return best_position, best_intensity

    def track_single_frame_2d(self, frame, start_center, search_radius_px, current_time, frame_id, reference_frame=None, track_foci=True):
        """Track FOCI on single 2D frame with nucleus tracking"""
        try:
            if len(frame.shape) == 3:
                frame_gray = frame[0] if frame.shape[0] in [1, 3] else frame.mean(axis=0)
            else:
                frame_gray = frame
            
            # Track nucleus if reference exists
            tracked_nucleus_mask = None
            nucleus_center = None
            
            if reference_frame is not None and self.reference_nucleus_mask is not None:
                tracked_nucleus_mask, nucleus_center = self.track_nucleus_across_frames(
                    reference_frame, frame_gray, self.reference_nucleus_mask
                )
                
                if tracked_nucleus_mask is not None:
                    self.total_roi_points = self.create_nucleus_roi_from_mask(tracked_nucleus_mask)
            
            # Track FOCI - только если track_foci=True
            if track_foci:
                foci_center, max_intensity = self.find_best_foci_position(
                    frame_gray, start_center, search_radius_px, self.foci_roi_radius
                )
            else:
                # Используем точно заданный центр без поиска
                foci_center = start_center
                max_intensity = self.measure_intensity_in_roi(frame_gray, foci_center, self.foci_roi_radius)
            
            # Measure intensities
            foci_intensity = self.measure_intensity_in_roi(frame_gray, foci_center, self.foci_roi_radius)
            background_intensity = self.measure_background_intensity_roi(frame_gray, self.background_roi_points)
            corrected_intensity = foci_intensity - background_intensity
            
            # Save tracking results
            frame_key = f"{frame_id}_{current_time}"
            self.tracked_results[frame_key] = {
                'foci_center': foci_center,
                'nucleus_center': nucleus_center,
                'nucleus_mask': tracked_nucleus_mask,
                'nucleus_roi': self.total_roi_points,
                'foci_intensity': foci_intensity,
                'background_intensity': background_intensity,
                'corrected_intensity': corrected_intensity
            }
            
            # Create result for CSV
            result = {
                'Time_s': current_time,
                'X_center': foci_center[0],
                'Y_center': foci_center[1],
                'Total_Intensity': foci_intensity,
                'Background_Intensity': background_intensity,
                'Corrected_Intensity': corrected_intensity,
                'Radius_px': self.foci_roi_radius,
                'Nucleus_Area': self.reference_area if hasattr(self, 'reference_area') else 0,
                'Max_Search_Intensity': max_intensity if track_foci else foci_intensity
            }
            
            # Create processed image
            processed_frame = self.create_processed_frame_2d(
                frame_gray, foci_center, search_radius_px, frame_id,
                nucleus_center, tracked_nucleus_mask
            )
            
            return [result], [processed_frame]
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None

    def create_processed_frame_2d(self, frame, foci_center, search_radius, frame_id, nucleus_center=None, nucleus_mask=None):
        """Create 2D frame with tracking visualization"""
        if len(frame.shape) == 2:
            display_frame = np.stack([frame] * 3, axis=-1)
        else:
            display_frame = frame.copy()
            if display_frame.shape[2] == 1:
                display_frame = np.stack([display_frame[:,:,0]] * 3, axis=-1)
        
        display_frame = (display_frame - display_frame.min()) / (display_frame.max() - display_frame.min())
        
        foci_x, foci_y = foci_center
        
        # Draw FOCI ROI outline (red) without fill
        rr, cc = draw.circle_perimeter(int(foci_y), int(foci_x), int(self.foci_roi_radius), shape=frame.shape[:2])
        valid_mask = (rr >= 0) & (rr < frame.shape[0]) & (cc >= 0) & (cc < frame.shape[1])
        display_frame[rr[valid_mask], cc[valid_mask], 0] = 1.0  # Red
        display_frame[rr[valid_mask], cc[valid_mask], 1] = 0.0
        display_frame[rr[valid_mask], cc[valid_mask], 2] = 0.0
        
        # Draw cross at FOCI center (yellow)
        cross_size = 5
        y_start = max(0, int(foci_y - cross_size))
        y_end = min(frame.shape[0], int(foci_y + cross_size))
        x_start = max(0, int(foci_x - cross_size))
        x_end = min(frame.shape[1], int(foci_x + cross_size))
        
        if y_start < y_end:
            display_frame[y_start:y_end, int(foci_x), :] = [1, 1, 0]
        if x_start < x_end:
            display_frame[int(foci_y), x_start:x_end, :] = [1, 1, 0]
        
        # Draw nucleus contour if exists
        if nucleus_mask is not None:
            from skimage import measure
            try:
                contours = measure.find_contours(nucleus_mask, 0.5)
                if contours:
                    main_contour = max(contours, key=len)
                    for i in range(len(main_contour) - 1):
                        y1, x1 = main_contour[i]
                        y2, x2 = main_contour[i + 1]
                        rr, cc = draw.line(int(y1), int(x1), int(y2), int(x2))
                        valid_mask = (rr >= 0) & (rr < frame.shape[0]) & (cc >= 0) & (cc < frame.shape[1])
                        display_frame[rr[valid_mask], cc[valid_mask], 1] = 1.0
            except Exception as e:
                pass
        
        # Draw nucleus center if exists
        if nucleus_center is not None:
            nuc_x, nuc_y = nucleus_center
            nuc_y, nuc_x = int(nuc_y), int(nuc_x)
            if 0 <= nuc_x < frame.shape[1] and 0 <= nuc_y < frame.shape[0]:
                display_frame[nuc_y-2:nuc_y+3, nuc_x-2:nuc_x+3, 1] = 1.0
        
        # Add text
        pil_image = Image.fromarray((display_frame * 255).astype(np.uint8))
        draw_obj = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        info_text = f"Frame: {frame_id}\nFOCI: ({foci_x:.1f}, {foci_y:.1f})\nR: {self.foci_roi_radius:.1f}px"
        draw_obj.text((10, 10), info_text, fill=(255, 255, 0), font=font)
        
        return np.array(pil_image)

    def save_results_2d(self, results, processed_frames, image_files, output_folder, resolution, search_radius, time_interval):
        """Save 2D analysis results"""
        print("💾 Saving results...")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get custom CSV name from user
        custom_name = simpledialog.askstring(
            "CSV Filename",
            f"Enter filename suffix (will be saved as {self.csv_base_name}_[suffix].csv):",
            initialvalue="experiment"
        )
        
        if custom_name:
            csv_filename = f"{self.csv_base_name}_{custom_name}.csv"
        else:
            csv_filename = f"{self.csv_base_name}_default.csv"
        
        final_output_folder = os.path.join(output_folder, f"FRAP_Analysis_2D_{timestamp}")
        os.makedirs(final_output_folder, exist_ok=True)
        
        if results:
            # Create DataFrame with time in first column and intensities in following columns
            df_data = []
            for result in results:
                row = {
                    'Time_s': result['Time_s'],
                    'Foci_Intensity': result['Corrected_Intensity'],
                    'Nucleus_Intensity': result.get('Total_Intensity', 0),
                    'Background_Intensity': result['Background_Intensity']
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            csv_path = os.path.join(final_output_folder, csv_filename)
            df.to_csv(csv_path, index=False)
            print(f"✅ Results saved: {csv_path}")
        else:
            print("⚠️ No data to save to CSV")
        
        if processed_frames:
            frames_folder = os.path.join(final_output_folder, "processed_frames")
            os.makedirs(frames_folder, exist_ok=True)
            
            for i, frame in enumerate(processed_frames):
                frame_path = os.path.join(frames_folder, f"frame_{i:04d}.png")
                io.imsave(frame_path, frame)
            
            print(f"✅ Processed frames saved: {frames_folder} ({len(processed_frames)} frames)")
        else:
            print("⚠️ No processed frames to save")
        
        self.create_analysis_report_2d(final_output_folder, results, image_files, resolution, search_radius, time_interval)
        
        return final_output_folder

    def create_analysis_report_2d(self, output_folder, results, image_files, resolution, search_radius, time_interval):
        """Create 2D analysis report"""
        print("📊 Creating analysis report...")
        
        report_path = os.path.join(output_folder, "analysis_report_2D.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("         FRAP-Tracker ADCANCED by IGB - 2D ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("📋 GENERAL INFORMATION:\n")
            f.write(f"   Analysis date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"   Resolution: {resolution} pixels/micron\n")
            f.write(f"   Search radius: {search_radius} microns\n")
            f.write(f"   Time interval: {time_interval} seconds\n")
            f.write(f"   FOCI ROI radius: {self.foci_roi_radius} pixels\n")
            f.write(f"   Total frames analyzed: {len(results) if results else 0}\n\n")
            
            f.write("📁 FILES USED:\n")
            for key, path in image_files.items():
                f.write(f"   {key}: {os.path.basename(path)}\n")
            f.write("\n")
            
            f.write("📊 ANALYSIS RESULTS:\n")
            if results:
                first_frame = results[0]
                last_frame = results[-1]
                
                f.write(f"   Initial FOCI position: ({first_frame['X_center']:.1f}, {first_frame['Y_center']:.1f})\n")
                f.write(f"   Final FOCI position: ({last_frame['X_center']:.1f}, {last_frame['Y_center']:.1f})\n")
                
                displacement = distance.euclidean(
                    [first_frame['X_center'], first_frame['Y_center']],
                    [last_frame['X_center'], last_frame['Y_center']]
                )
                displacement_um = displacement / resolution
                
                f.write(f"   Total displacement: {displacement:.1f} pixels ({displacement_um:.1f} microns)\n")
                
                intensities = [r['Corrected_Intensity'] for r in results]
                f.write(f"   Maximum intensity: {max(intensities):.1f}\n")
                f.write(f"   Minimum intensity: {min(intensities):.1f}\n")
                f.write(f"   Mean intensity: {np.mean(intensities):.1f}\n")
                
                # FRAP kinetics analysis
                pre_intensities = [r['Corrected_Intensity'] for r in results if 'PRE' in str(r.get('Frame_ID', ''))]
                bleach_intensity = [r['Corrected_Intensity'] for r in results if 'BLEACH' in str(r.get('Frame_ID', ''))]
                post_intensities = [r['Corrected_Intensity'] for r in results if 'POST' in str(r.get('Frame_ID', ''))]
                
                if pre_intensities and bleach_intensity and post_intensities:
                    avg_pre = np.mean(pre_intensities)
                    bleach_val = bleach_intensity[0] if bleach_intensity else 0
                    max_post = max(post_intensities) if post_intensities else 0
                    
                    f.write(f"   Mean PRE intensity: {avg_pre:.1f}\n")
                    f.write(f"   BLEACH intensity: {bleach_val:.1f}\n")
                    f.write(f"   Maximum POST intensity: {max_post:.1f}\n")
                    if avg_pre > 0:
                        recovery_ratio = (max_post - bleach_val) / (avg_pre - bleach_val) * 100
                        f.write(f"   FRAP recovery: {recovery_ratio:.1f}%\n")
            else:
                f.write("   No data for analysis\n")
        
        print(f"✅ Report saved: {report_path}")

    def run_analysis_2d(self):
        """MAIN FUNCTION: Run complete 2D FRAP analysis"""
        try:
            print("=== 🚀 STARTING FRAP-Tracker ADCANCED by IGB (2D VERSION) ===")
            
            # ==================== BLOCK 1: GET PARAMETERS ====================
            print("\n=== 📋 BLOCK 1: GET PARAMETERS ===")
            resolution = simpledialog.askfloat("Resolution", "Enter pixels per micron?")
            if resolution is None:
                return
            print(f"✅ Resolution: {resolution} pixels/micron")
            
            search_radius = simpledialog.askfloat(
                "Search radius", 
                "Enter FOCI search radius in microns:\n\n"
                "💡 Explanation: Maximum distance FOCI can move between frames.\nTypically 1-5 microns."
            )
            if search_radius is None:
                return
            print(f"✅ Search radius: {search_radius} microns")
                
            time_interval = simpledialog.askfloat("Time interval", 
                                                "Enter interval between frames (seconds):")
            if time_interval is None:
                return
            print(f"✅ Time interval: {time_interval} seconds")
            
            # ==================== BLOCK 2: LOAD 2D IMAGES ====================
            print("\n=== 🖼️ BLOCK 2: LOAD 2D IMAGES ===")
            image_files = self.load_image_files()
            if not image_files:
                return
            
            print("Loading PRE BLEACH image...")
            pre_bleach_image = io.imread(image_files['PRE_BLEACH'])
            
            print("Loading BLEACH image...")
            bleach_image = io.imread(image_files['BLEACH'])
            
            print("Loading POST BLEACH image...")
            post_bleach_image = io.imread(image_files['POST_BLEACH'])
            
            print("Loading MASK image...")
            mask_image = io.imread(image_files['MASK_BLEACH'])
            
            # ==================== BLOCK 3: AUTOMATIC MASK BINARIZATION ====================
            print("\n=== 🎨 BLOCK 3: AUTOMATIC MASK BINARIZATION ===")
            binary_mask = self.create_and_validate_mask(mask_image)
            if binary_mask is None:
                print("❌ Mask binarization cancelled by user")
                return
            
            # ==================== BLOCK 4: DETECT ROI FROM BINARY MASK ====================
            print("\n=== 🎯 BLOCK 4: DETECT ROI FROM BINARY MASK ===")
            roi_center, roi_radius = self.detect_roi_from_binary_mask(binary_mask)
            
            # ==================== BLOCK 5: SELECT FOCI ROI ON FIRST PRE BLEACH FRAME ====================
            print("\n=== 🎯 BLOCK 5: SELECT FOCI ROI ON FIRST PRE BLEACH FRAME ===")
            
            # Get first PRE BLEACH frame
            if len(pre_bleach_image.shape) == 3:
                pre_first_frame = pre_bleach_image[0]
            else:
                pre_first_frame = pre_bleach_image
                
            foci_selection_success = self.select_foci_roi_on_first_pre_bleach(pre_first_frame, bleach_image)
            
            if not foci_selection_success or not self.foci_roi_center_first_pre or not self.foci_roi_radius:
                print("❌ FOCI ROI selection cancelled or failed")
                return
            
            # Convert search radius to pixels
            search_radius_px = search_radius * resolution
            
            # ==================== BLOCK 6: MANUAL NUCLEUS SELECTION ON BLEACH IMAGE ====================
            print("\n=== 🎯 BLOCK 6: MANUAL NUCLEUS SELECTION ===")
            selection_success = self.manual_nucleus_selection_2d(bleach_image)
            
            if not selection_success:
                print("❌ Manual nucleus selection cancelled or failed")
                return
            
            self.total_roi_points = self.create_nucleus_roi_from_mask(self.reference_nucleus_mask)
            
            # ==================== BLOCK 7: SELECT BACKGROUND ROI WITH SLIDER ====================
            print("\n=== 🔵 BLOCK 7: SELECT BACKGROUND ROI WITH SLIDER ===")
            
            background_stack = []
            if len(pre_bleach_image.shape) == 3:
                background_stack.extend([pre_bleach_image[i] for i in range(pre_bleach_image.shape[0])])
            else:
                background_stack.append(pre_bleach_image)
            
            if len(bleach_image.shape) == 3:
                background_stack.extend([bleach_image[i] for i in range(bleach_image.shape[0])])
            else:
                background_stack.append(bleach_image)
                
            if len(post_bleach_image.shape) == 3:
                background_stack.extend([post_bleach_image[i] for i in range(post_bleach_image.shape[0])])
            else:
                background_stack.append(post_bleach_image)
            
            background_success = self.select_background_roi_with_slider(background_stack)
            
            if not background_success or not self.background_roi_points:
                print("❌ BACKGROUND ROI selection cancelled")
                return
            
            # ==================== BLOCK 8: SELECT FOCI POSITION ON BLEACH WITH SLIDER ====================
            print("\n=== 🔴 BLOCK 8: SELECT FOCI POSITION ON BLEACH WITH SLIDER ===")
            
            foci_stack = []
            if len(bleach_image.shape) == 3:
                foci_stack.extend([bleach_image[i] for i in range(bleach_image.shape[0])])
            else:
                foci_stack.append(bleach_image)
            
            foci_success = self.select_foci_position_with_slider(foci_stack)
            
            if not foci_success or not self.selected_bleach_center:
                print("❌ FOCI selection on BLEACH cancelled")
                return
            
            # ==================== BLOCK 9: TRACK FOCI AND NUCLEUS ACROSS ALL FRAMES ====================
            print("\n=== 🎯 BLOCK 9: TRACKING FOCI AND NUCLEUS ACROSS ALL FRAMES ===")
            all_results = []
            all_processed_frames = []
            
            # Reference image for tracking (BLEACH)
            reference_frame = bleach_image
            if len(reference_frame.shape) == 3:
                reference_frame = reference_frame[0] if reference_frame.shape[0] in [1, 3] else reference_frame.mean(axis=0)
            
            # ==================== PRE BLEACH TRACKING (В ОБРАТНУЮ СТОРОНУ) ====================
            print("\n--- 🔄 TRACKING PRE BLEACH (BACKWARDS FROM LAST TO FIRST) ---")
            if len(pre_bleach_image.shape) == 3:
                # Начинаем с последнего PRE кадра, используя центр с первого PRE как ориентир
                current_foci_center = self.foci_roi_center_first_pre
                n_pre = pre_bleach_image.shape[0]
                
                # Сначала находим позицию на последнем PRE кадре, идя от первого
                print(f"   Finding position on last PRE frame (PRE_{n_pre-1})...")
                temp_results, _ = self.track_single_frame_2d(
                    pre_bleach_image[n_pre-1], current_foci_center, search_radius_px, 
                    0, f"TEMP", reference_frame, track_foci=True
                )
                if temp_results:
                    last_pre_center = (temp_results[0]['X_center'], temp_results[0]['Y_center'])
                    print(f"   Last PRE frame center: {last_pre_center}")
                    
                    # Теперь идем ОБРАТНО от последнего к первому
                    current_foci_center = last_pre_center
                    for i in range(n_pre-1, -1, -1):
                        pre_time = time_interval * i
                        print(f"   Tracking PRE frame {i} (backwards)...")
                        
                        pre_results, pre_processed = self.track_single_frame_2d(
                            pre_bleach_image[i], current_foci_center, search_radius_px, 
                            pre_time, f"PRE_{i}", reference_frame, track_foci=True
                        )
                        
                        if pre_results:
                            all_results.insert(0, pre_results[0])  # Вставляем в начало для правильного порядка времени
                            all_processed_frames.insert(0, pre_processed[0])
                            current_foci_center = (pre_results[0]['X_center'], pre_results[0]['Y_center'])
                            print(f"      Found: {current_foci_center}")
            else:
                # Если только один кадр PRE
                pre_time = 0
                pre_results, pre_processed = self.track_single_frame_2d(
                    pre_bleach_image, self.foci_roi_center_first_pre, search_radius_px,
                    pre_time, "PRE_0", reference_frame, track_foci=True
                )
                if pre_results:
                    all_results.append(pre_results[0])
                    all_processed_frames.append(pre_processed[0])
            
            # ==================== BLEACH TRACKING (ТОЧНО КАК ВЫБРАЛ ПОЛЬЗОВАТЕЛЬ) ====================
            print("\n--- 🔥 TRACKING BLEACH (USING EXACT USER SELECTION) ---")
            if len(pre_bleach_image.shape) == 3:
                bleach_time = time_interval * pre_bleach_image.shape[0]
            else:
                bleach_time = time_interval
            
            # Важно: track_foci=False - используем точно выбранный центр без поиска
            bleach_results, bleach_processed = self.track_single_frame_2d(
                bleach_image, self.bleach_foci_center, search_radius_px, 
                bleach_time, "BLEACH", reference_frame, track_foci=False
            )
            if bleach_results:
                all_results.append(bleach_results[0])
                all_processed_frames.append(bleach_processed[0])
                print(f"   Using exact user position: {self.bleach_foci_center}")
            
            # ==================== POST BLEACH TRACKING (ОТ ПЕРВОГО К ПОСЛЕДНЕМУ) ====================
            print("\n--- ⏩ TRACKING POST BLEACH (FORWARDS) ---")
            if len(post_bleach_image.shape) == 3:
                current_foci_center = self.bleach_foci_center  # Начинаем с точно выбранной позиции на BLEACH
                
                for i in range(post_bleach_image.shape[0]):
                    post_time = bleach_time + time_interval * (i + 1)
                    post_frame = post_bleach_image[i]
                    
                    print(f"   Tracking POST frame {i}...")
                    post_results, post_processed = self.track_single_frame_2d(
                        post_frame, current_foci_center, search_radius_px, 
                        post_time, f"POST_{i}", reference_frame, track_foci=True
                    )
                    
                    if post_results:
                        all_results.append(post_results[0])
                        all_processed_frames.append(post_processed[0])
                        current_foci_center = (post_results[0]['X_center'], post_results[0]['Y_center'])
                        print(f"      Found: {current_foci_center}")
            else:
                post_time = bleach_time + time_interval
                post_results, post_processed = self.track_single_frame_2d(
                    post_bleach_image, self.bleach_foci_center, search_radius_px, 
                    post_time, "POST_0", reference_frame, track_foci=True
                )
                if post_results:
                    all_results.append(post_results[0])
                    all_processed_frames.append(post_processed[0])
            
            # ==================== BLOCK 10: SELECT OUTPUT FOLDER ====================
            print("\n=== 📁 BLOCK 10: SELECT OUTPUT FOLDER ===")
            messagebox.showinfo("Save results", "Select folder to save analysis results")
            output_folder = filedialog.askdirectory(title="Select folder to save results")
            if not output_folder:
                original_dir = os.path.dirname(image_files['PRE_BLEACH'])
                output_folder = os.path.join(original_dir, "FRAP_Tracker_IGB_Results_2D")
                print(f"⚠ No folder selected, using default: {output_folder}")
            else:
                output_folder = os.path.join(output_folder, "FRAP_Tracker_IGB_Results_2D")
                print(f"✅ Selected output folder: {output_folder}")
            
            # ==================== BLOCK 11: SAVE RESULTS ====================
            print("\n=== 💾 BLOCK 11: SAVING RESULTS ===")
            final_output_folder = self.save_results_2d(all_results, all_processed_frames, image_files, output_folder, resolution, search_radius, time_interval)
            
            print(f"\n=== 🎉 2D FRAP ANALYSIS COMPLETE ===")
            print(f"✅ Total frames processed: {len(all_results)}")
            print(f"✅ Frames with outlines saved: {len(all_processed_frames)}")
            print(f"✅ Results saved to: {final_output_folder}")
            
            messagebox.showinfo("Complete", 
                              f"FRAP-Tracker ADCANCED by IGB completed!\n\n"
                              f"🎉 Frames processed: {len(all_results)}\n"
                              f"🖼️ Frames with outlines: {len(all_processed_frames)}\n"
                              f"💾 Results saved to:\n{final_output_folder}")
            
        except Exception as e:
            error_msg = f"Error occurred: {str(e)}"
            print(f"❌ {error_msg}")
            messagebox.showerror("Error", error_msg)
            import traceback
            traceback.print_exc()

# ==================== PROGRAM START ====================
if __name__ == "__main__":
    print("🧪 Initializing FRAP-Tracker ADCANCED by IGB (2D VERSION)...")
    print("===========================================")
    analyzer = FRAPTrackerIGB()
    analyzer.run_analysis_2d()