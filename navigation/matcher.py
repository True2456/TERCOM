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
