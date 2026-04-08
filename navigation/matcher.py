import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift

class FFTMatcher:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
        
    def match(self, live_frame, map_tile, denoise=False, edge_match=False, native_size=False):
        """
        Performs Phase Correlation between live frame and map tile.
        Both images should be pre-scaled and orthorectified.
        native_size=True: skip internal resize (use when tile is already small,
                          e.g. 52x39 px — avoids destroying texture by 10x upscale).
        """
        if native_size:
            # Use inputs as-is; both must already be the same size
            h, w = live_frame.shape[:2]
            live   = live_frame.astype(np.float32)
            m_tile = map_tile.astype(np.float32)
            work_h, work_w = h, w
        else:
            # Ensure identical sizes via resize
            live   = cv2.resize(live_frame, self.target_size)
            m_tile = cv2.resize(map_tile,   self.target_size)
            work_h, work_w = self.target_size[1], self.target_size[0]

        # Cross-Modal Alignment: Blurring to match map resolution
        if denoise:
            live = cv2.GaussianBlur(live, (11, 11), 0)

        if edge_match:
            l_u8   = cv2.normalize(live,   None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            m_u8   = cv2.normalize(m_tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Structural edge extraction: Blur out organic foliage noise
            l_blur = cv2.GaussianBlur(l_u8, (5, 5), 0)
            m_blur = cv2.GaussianBlur(m_u8, (5, 5), 0)
            
            # Extract sharp edges (roads, roofs)
            l_edge = cv2.Canny(l_blur, 50, 150)
            m_edge = cv2.Canny(m_blur, 50, 150)

            # Thicken structural vectors to grant them dominant Phase mass
            kernel = np.ones((3, 3), np.uint8)
            live   = cv2.dilate(l_edge, kernel, iterations=1).astype(np.float32)
            m_tile = cv2.dilate(m_edge, kernel, iterations=1).astype(np.float32)

        # Windowing to avoid edge effects (Hanning)
        win    = np.hanning(work_h)[:, None] * np.hanning(work_w)
        live   = live.astype(np.float32)   * win
        m_tile = m_tile.astype(np.float32) * win
        
        # FFT Phase Correlation
        f1 = fft2(live)
        f2 = fft2(m_tile)
        
        cross_power = (f1 * f2.conj()) / (np.abs(f1 * f2.conj()) + 1e-12)
        r = ifft2(cross_power)
        r = np.abs(fftshift(r))
        
        # Find peak
        max_val = np.max(r)
        y, x = np.unravel_index(r.argmax(), r.shape)
        
        # Calculate Peak-to-Sidelobe Ratio (PSR) for confidence
        # PSR = (max - mean) / std
        mean = np.mean(r)
        std = np.std(r)
        psr = (max_val - mean) / (std + 1e-12)
        
        # Centering (displacement from middle of shifted r)
        center_y, center_x = work_h // 2, work_w // 2
        dy = y - center_y
        dx = x - center_x
        
        return dx, dy, psr

    def orthorectify(self, img, roll, pitch, alt_agl, h_fov=110):
        """
        Corrects image for attitude using perspective warp.
        Assumes nadir camera looking straight down if pitch/roll=0.
        """
        h, w = img.shape
        f = (w / 2) / np.tan(np.deg2rad(h_fov / 2))
        
        # Camera matrix
        K = np.array([
            [f, 0, w/2],
            [0, f, h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Rotation Matrices
        r_rad = np.deg2rad(roll)
        p_rad = np.deg2rad(pitch)
        
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(p_rad), -np.sin(p_rad)],
            [0, np.sin(p_rad), np.cos(p_rad)]
        ])
        
        R_roll = np.array([
            [np.cos(r_rad), 0, np.sin(r_rad)],
            [0, 1, 0],
            [-np.sin(r_rad), 0, np.cos(r_rad)]
        ])
        
        R = R_roll @ R_pitch
        
        # Homography H = K * R * K^-1
        H = K @ R @ np.linalg.inv(K)
        
        return cv2.warpPerspective(img, H, (w, h))

class ORBMatcher:
    def __init__(self, max_features=1000):
        self.orb = cv2.ORB_create(max_features)
        # NORM_HAMMING is required for ORB descriptor comparison
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def match(self, live_frame, map_tile, native_size=True):
        """
        Extracts keypoints using ORB and returns the best affine shift mapping
        the center of live_frame to map_tile.
        """
        # 1. Feature Extraction
        kp1, des1 = self.orb.detectAndCompute(live_frame, None)
        kp2, des2 = self.orb.detectAndCompute(map_tile, None)
        
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return 0.0, 0.0, 0.0  # Failed to match enough features
            
        # 2. Descriptor Matching
        matches = self.matcher.match(des1, des2)
        # Sort them in the order of their distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # We need at least 4 good matches to find Homography
        if len(matches) < 4:
            return 0.0, 0.0, 0.0
            
        # Extract location of good matches
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 3. Find Homography (Perspective Transform)
        matrix, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        
        if matrix is None:
            return 0.0, 0.0, 0.0
            
        # The number of inliers (matches that fit the physical geometry) acts as our 'confidence' score
        # replacing the concept of a 'PSR' in Phase Correlation.
        inliers = np.sum(mask)
        
        # 4. Map the center of the drone photo to the map tile
        h, w = live_frame.shape[:2]
        center_pt = np.float32([[[w / 2.0, h / 2.0]]])
        transformed_center = cv2.perspectiveTransform(center_pt, matrix)
        
        tc_x, tc_y = transformed_center[0][0]
        
        # To make it compatible with the exact outputs expected by the `trn_refinement_optical.py` slider:
        # Instead of sliding the window physically, we are comparing the photo directly against the ENTIRE patch.
        # But wait, trn_refinement_optical.py expects dr, dc offset of the top-left corner relative to the sliding window step.
        # Let's adjust trn_refinement_optical.py to NOT use a sliding window if ORB is used!
        # ORB inherently compares the photo against the full map tile natively without sliding windows!
        # So we can just return the center offset coordinate mapping!
        
        return float(tc_x), float(tc_y), float(inliers)

if __name__ == "__main__":
    # Test with synthetic offset
    matcher = FFTMatcher()
    img = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(img, (200, 200), (300, 300), 255, -1)
    
    # Create shifted version
    M = np.float32([[1, 0, 15], [0, 1, -10]]) # 15 px right, 10 px up
    shifted = cv2.warpAffine(img, M, (512, 512))
    
    dx, dy, psr = matcher.match(shifted, img)
    print(f"Shift: dx={dx}, dy={dy}, Confidence (PSR): {psr:.2f}")
