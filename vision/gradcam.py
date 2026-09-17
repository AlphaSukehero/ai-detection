"""Saliency maps and the tumour geometry derived from them.

Split out of app.py: none of this touches Flask, and the geometry functions
below are read as measurements in the PDF report, so they belong somewhere
they can be tested on their own.

A caveat that belongs with the code rather than in a commit message: the
"tumour" area, size and location are computed from a thresholded Grad-CAM
heatmap. Grad-CAM is an explainability artefact showing where a classifier
looked -- it is not a segmentation, and these numbers inherit that. They are
labelled "Estimated" everywhere they surface for that reason.
"""
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except Exception as e:  # pragma: no cover - environment dependent
    print("OpenCV import warning:", e)
    CV2_AVAILABLE = False


def generate_gradcam_heatmap(model, image_array, predicted_class):
    """Generate a Grad-CAM heatmap for the predicted class."""
    try:
        import tensorflow as tf
        
        # Recursively find the last conv layer (including inside Functional sub-models like vgg16)
        def find_last_conv(layer):
            # If this layer has sub-layers, recurse into them
            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                for sub in reversed(layer.layers):
                    result = find_last_conv(sub)
                    if result is not None:
                        return result
            # Check if this is a Conv2D layer
            if 'conv' in layer.__class__.__name__.lower():
                return layer
            return None
        
        # Find the inner model that contains the conv layers
        inner_model = model
        last_conv_layer = None
        
        for layer in reversed(model.layers):
            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                # This is a Functional sub-model (like vgg16)
                for sub in reversed(layer.layers):
                    conv = find_last_conv(sub)
                    if conv is not None:
                        last_conv_layer = conv
                        inner_model = layer
                        break
            else:
                conv = find_last_conv(layer)
                if conv is not None:
                    last_conv_layer = conv
                    break
        
        if last_conv_layer is None:
            return generate_gradient_heatmap(model, image_array, predicted_class)
        
        # Build a sub-model from the inner model's input to the conv layer and output
        # The inner model like vgg16 has its own input
        grad_model = tf.keras.models.Model(
            inputs=inner_model.input,
            outputs=[last_conv_layer.output, inner_model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array)
            loss = predictions[:, predicted_class]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            return generate_gradient_heatmap(model, image_array, predicted_class)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize
        heatmap_max = tf.reduce_max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        
        # Convert to numpy
        heatmap_np = heatmap.numpy()
        
        # Resize to 224x224
        if cv2 is not None:
            heatmap_np = cv2.resize(heatmap_np, (224, 224))
        else:
            from PIL import Image as PILImage
            heatmap_img = PILImage.fromarray((heatmap_np * 255).astype(np.uint8))
            heatmap_img = heatmap_img.resize((224, 224), PILImage.BILINEAR)
            heatmap_np = np.array(heatmap_img) / 255.0
        
        return heatmap_np
        
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        return generate_gradient_heatmap(model, image_array, predicted_class)


def generate_gradient_heatmap(model, image_array, predicted_class):
    """Fallback: Generate gradient-based heatmap when Grad-CAM fails."""
    try:
        import tensorflow as tf
        
        # Ensure the input array has the right shape for the model (Batch, H, W, C)
        if image_array.ndim == 3:
            input_tensor = tf.convert_to_tensor(image_array[None, ...])
        else:
            input_tensor = tf.convert_to_tensor(image_array)
        
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = model(input_tensor)
            loss = predictions[:, predicted_class]
        
        grads = tape.gradient(loss, input_tensor)
        
        if grads is None:
            return None
        
        # Take absolute value and mean across channels
        grads_np = grads.numpy()[0]
        if grads_np.ndim == 3:
            heatmap = np.mean(np.abs(grads_np), axis=-1)
        else:
            heatmap = np.abs(grads_np)
        
        # Normalize
        heatmap_max = heatmap.max()
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        
        return heatmap
        
    except Exception as e:
        print(f"Gradient heatmap generation failed: {e}")
        return None


def create_gradcam_overlay(original_image, heatmap, alpha=0.45):
    """Create a heatmap overlay on the original image."""
    if heatmap is None:
        return original_image
    
    # Convert original image to numpy array
    original_np = np.array(original_image.resize((224, 224)))
    
    # Apply colormap to heatmap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    if cv2 is not None:
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    else:
        # Fallback without OpenCV
        heatmap_colored = np.stack([heatmap_uint8] * 3, axis=-1)
        # Apply red channel emphasis
        heatmap_colored[:, :, 0] = heatmap_uint8
        heatmap_colored[:, :, 1] = 0
        heatmap_colored[:, :, 2] = 255 - heatmap_uint8
    
    # Blend
    overlay = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0) if cv2 is not None else \
        (original_np * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    
    return Image.fromarray(overlay)


def create_tumor_region_highlight(original_image, heatmap, threshold_percentile=90):
    """Create a highlighted tumor region image."""
    if heatmap is None:
        return original_image
    
    original_np = np.array(original_image.resize((224, 224)))
    
    # Threshold the heatmap
    threshold = np.percentile(heatmap, threshold_percentile)
    mask = (heatmap >= threshold).astype(np.uint8) * 255
    
    # Clean up mask
    if cv2 is not None:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Create red overlay
    overlay = original_np.copy()
    overlay[mask > 0] = [255, 0, 0]  # Red highlight
    
    # Blend
    highlighted = cv2.addWeighted(original_np, 0.65, overlay, 0.35, 0) if cv2 is not None else \
        (original_np * 0.65 + overlay * 0.35).astype(np.uint8)
    
    # Draw contour if OpenCV available
    if cv2 is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(highlighted, [largest_contour], -1, (255, 255, 0), 2)  # Yellow boundary
    
    return Image.fromarray(highlighted)


def calculate_tumor_area(mask):
    """Calculate tumor area as percentage of image."""
    active_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]
    return float(active_pixels / total_pixels * 100)


def calculate_tumor_size(contour):
    """Calculate tumor bounding box size."""
    if contour is None or not CV2_AVAILABLE:
        return 0, 0
    x, y, width, height = cv2.boundingRect(contour)
    return int(width), int(height)


def calculate_tumor_location(contour):
    """Calculate tumor location description."""
    if contour is None or not CV2_AVAILABLE:
        return "Not available"
    
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return "Not available"
    
    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"]
    
    horizontal = "Left" if center_x < 74 else ("Central" if center_x < 150 else "Right")
    vertical = "Upper" if center_y < 74 else ("Middle" if center_y < 150 else "Lower")
    
    return f"{horizontal} {vertical} region"


def calculate_severity(area):
    """Calculate severity based on area."""
    if area < 5:
        return "Low"
    elif area < 15:
        return "Moderate"
    else:
        return "High"


def calculate_spread(area):
    """Calculate spread based on area."""
    if area < 5:
        return "Limited"
    elif area < 15:
        return "Moderate"
    else:
        return "Extensive"


