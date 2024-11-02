import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


def non_maximum_suppression(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression on boxes with corresponding scores.
    
    Args:
        boxes: numpy array of shape (N, 4) containing bounding boxes in format [x1, y1, x2, y2]
        scores: numpy array of shape (N,) containing confidence scores
        iou_threshold: float, threshold for IoU to determine overlap
        
    Returns:
        numpy array: indices of boxes to keep
    """
    
    # If no boxes, return empty list
    if len(boxes) == 0:
        return []
    
    # Convert boxes to floats
    boxes = boxes.astype(float)
    
    # Get coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate area of bounding boxes
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by confidence score
    order = scores.argsort()[::-1]
    
    keep = []  # List to store indices of kept boxes
    
    while order.size > 0:
        # Pick the box with highest confidence score
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        # Get coordinates of intersection
        xx1 = np.maximum(x1[i], x1[order[1:]])
        print(f'xx1 shape: {xx1.shape}')
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # Calculate intersection area
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        # print(f'intersection: {intersection}')
        
        # Calculate union area
        union = areas[i] + areas[order[1:]] - intersection
        # print(f'union: {union}')
        
        # Calculate IoU
        iou = intersection / union
        # print(f'iou: {iou}')
        
        # Keep boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep)


def test_nms():
    """
    Test suite for Non-Maximum Suppression implementation.
    """
    
    def test_empty_input():
        """Test behavior with empty input."""
        boxes = np.array([])
        scores = np.array([])
        result = non_maximum_suppression(boxes, scores, 0.5)
        assert len(result) == 0, "Empty input should return empty output"
        
    def test_single_box():
        """Test with a single box."""
        boxes = np.array([[0, 0, 10, 10]])
        scores = np.array([0.9])
        result = non_maximum_suppression(boxes, scores, 0.5)
        assert_array_equal(result, [0]), "Single box should always be kept"
        
    def test_no_overlap():
        """Test with non-overlapping boxes."""
        boxes = np.array([
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50]
        ])
        scores = np.array([0.9, 0.8, 0.7])
        result = non_maximum_suppression(boxes, scores, 0.5)
        assert_array_equal(result, [0, 1, 2]), "Non-overlapping boxes should all be kept"
        
    def test_high_overlap():
        """Test with highly overlapping boxes."""
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 9, 9],
        ])
        scores = np.array([0.9, 0.8])
        result = non_maximum_suppression(boxes, scores, 0.5)
        assert_array_equal(result, [0]), "Only highest scoring box should be kept with high overlap"
        
    def test_partial_overlap():
        """Test with partially overlapping boxes."""
        boxes = np.array([
            [0, 0, 10, 10],
            [5, 5, 15, 15],
            [12, 12, 20, 20]
        ])
        scores = np.array([0.9, 0.8, 0.7])
        result = non_maximum_suppression(boxes, scores, 0.5)
        expected = [0, 2]  # First and last box should be kept
        assert_array_equal(result, expected), "Partially overlapping boxes should be handled correctly"
        
    def test_different_iou_thresholds():
        """Test behavior with different IoU thresholds."""
        boxes = np.array([
            [0, 0, 10, 10],
            [2, 2, 12, 12]
        ])
        scores = np.array([0.9, 0.8])
        
        # With low IoU threshold
        result_low = non_maximum_suppression(boxes, scores, 0.3)
        assert len(result_low) == 1, "Low IoU threshold should suppress more boxes"
        
        # With high IoU threshold
        result_high = non_maximum_suppression(boxes, scores, 0.7)
        assert len(result_high) == 2, "High IoU threshold should keep more boxes"
        
    def test_exact_overlap():
        """Test with exactly overlapping boxes."""
        boxes = np.array([
            [0, 0, 10, 10],
            [0, 0, 10, 10]
        ])
        scores = np.array([0.9, 0.8])
        result = non_maximum_suppression(boxes, scores, 0.5)
        assert_array_equal(result, [0]), "Exactly overlapping boxes should keep highest score"
        
    def test_different_box_sizes():
        """Test with boxes of different sizes."""
        boxes = np.array([
            [0, 0, 10, 10],    # Small box
            [0, 0, 20, 20],    # Large box
            [30, 30, 35, 35]   # Different small box
        ])
        scores = np.array([0.9, 0.8, 0.7])
        result = non_maximum_suppression(boxes, scores, 0.5)
        expected = [0, 2]  # Should keep highest scoring box and non-overlapping box
        assert_array_equal(result, expected), "Should handle different sized boxes correctly"

    # Run all tests
    #test_empty_input()
    #test_single_box()
    test_no_overlap()
    #test_high_overlap()
    #test_partial_overlap()
    #test_different_iou_thresholds()
    #test_exact_overlap()
    #test_different_box_sizes()
    
    print("All tests passed!")


# Run the test suite
if __name__ == "__main__":
    test_nms()