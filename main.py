import cv2

# Load the original image
image = cv2.imread("test_document.jpg")

# Set scale for displaying the images
scale = 0.25

# Calculate display dimensions
width = int(image.shape[1] * scale)
height = int(image.shape[0] * scale)

# Resize original image for display
display_image = cv2.resize(image, (width, height))

# Convert original image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize grayscale image for display
gray_display = cv2.resize(gray_image, (width, height))

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Resize blurred image for display
blurred_display = cv2.resize(blurred_image, (width, height))

# Apply Canny edge detection
edges = cv2.Canny(blurred_image, 50, 150)

# Resize edge image for display
edges_display = cv2.resize(edges, (width, height))

# Find contours in the edge image
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Sort contours from largest to smallest
contours = sorted(
    contours,
    key=cv2.contourArea,
    reverse=True
)

# Create a copy of the original image
contour_image = image.copy()

# Draw the largest contour
if contours:
    cv2.drawContours(
        contour_image,
        [contours[0]],
        -1,
        (0, 255, 0),
        10
    )

# Resize contour result for display
contour_display = cv2.resize(
    contour_image,
    (width, height)
)

# Display all processing stages
cv2.imshow("Original Image", display_image)
cv2.imshow("Grayscale Image", gray_display)
cv2.imshow("Blurred Image", blurred_display)
cv2.imshow("Edges", edges_display)
cv2.imshow("Largest Contour", contour_display)

# Wait until a key is pressed
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
