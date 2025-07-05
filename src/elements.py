import numpy as np

class Element:
    def __init__(self, name, position, width, present=True):
        self.name = name
        self._position = position
        self.width = width
        self.left = position[0] - width // 2
        self.right = position[0] + width // 2
        self.present = present

    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, value):
        self._position = value
        self.left = self._position[0] - self.width // 2
        self.right = self._position[0] + self.width // 2
    
    def is_aligned(self, element, margin = 0, tolerance = 0):
        selfLeft = self.left - margin
        selfRight = self.right + margin
        elementLeft = element.left + tolerance
        elementRight = element.right - tolerance
        return selfLeft <= elementLeft <= selfRight or selfLeft <= elementRight <= selfRight
    
    def x_diff(self, element):
        return element.position[0] - self.position[0]
    
    def y_diff(self, element):
        return self.position[1] - element.position[1]
    
    def is_same (self, element):
        return self.name == element.name and abs(self.position[0] - element.position[0]) < 5 and abs(self.position[1] - element.position[1]) < 5
    
    def __str__(self):
        return f"{self.name} {self.position}"


class Player (Element):
    color = [(20, 100, 100), (30, 255, 255)]
    area = [200, 400] 

    def __init__(self, position = [0, 0]):
        super().__init__("Player", position, 20, present=(position != [0,0]))
        self.position = position
        self.can_move_left = True
        self.can_move_right = True

    def is_aiming (self, element, tolerance = 2):
        selfCenter = self.position[0]
        elementCenter = element.position[0]
        elementLeft = elementCenter - tolerance
        elementRight = elementCenter + tolerance
        return elementLeft <= selfCenter <= elementRight


class Enemy (Element):
    def __init__(self, name, position, width):
        super().__init__(name, position, width)
        self.is_moving = False
        


class Helicopter (Enemy):
    color = [(50, 100, 50), (90, 255, 255)] # Dark Green
    area = [50, 53]

    def __init__(self, position):
        x, y = position
        super().__init__("Helicopter", [x-2, y], width=30)

class Boat (Enemy):
    color = [(0, 180, 150), (10, 255, 255)] # Dark Red
    area = [200, 250]

    def __init__(self, position):
        super().__init__("Boat", position, width=50)

class Plane (Enemy):
    color = [(100, 50, 100), (140, 150, 255)] # Light Blue
    area = [120, 130]

    def __init__(self, position):
        self.predicted_x_at_y0 = 0
        self.ground_y = 450  # Bottom of the screen (reversed y-axis)

        super().__init__("Plane", position, width=25)

        self.dt = 1.0

        # Initial Kalman state: [x, y, vx, vy]
        self.x_est = np.array([[position[0]],  # x
                               [position[1]],  # y
                               [0.0],          # vx
                               [0.0]])         # vy

        # State transition matrix
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Measurement matrix: we observe [x, y]
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Initial uncertainty
        self.P = np.eye(4)

        # No process or measurement noise
        self.Q = np.zeros((4, 4))
        self.R = np.zeros((2, 2))

    # KALMAN FILTER TO PREDICT PLANE POSITION
    @Element.position.setter
    def position(self, position):
        # Update the actual position
        super(Plane, Plane).position.__set__(self, position)

        # New measurement
        z = np.array([[position[0]],
                      [position[1]]])

        # Predict
        x_pred = self.A @ self.x_est
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # Update
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.pinv(S)

        self.x_est = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        # Predict X where Y == ground_y
        x_now, y_now, vx, vy = self.x_est.flatten()
        if abs(vy) > 1e-8:
            t_to_ground = (self.ground_y - y_now) / vy
            if t_to_ground > 0:
                x_at_y0 = x_now + vx * t_to_ground
            else:
                x_at_y0 = np.nan  # Already hit ground or going up
        else:
            x_at_y0 = np.nan  # No vertical velocity

        self.predicted_x_at_y0 = x_at_y0 % 455


class Fuel (Element):
    color = [(0, 100, 100), (5, 255, 255)] # Light Red
    area = [229.5, 229.5]

    def __init__(self, position):
        super().__init__("Fuel", position, width=20)


class Passing (Element):
    def __init__(self, start, end):
        super().__init__("Passing", [(end + start) // 2, 10], end - start)
    
    def includes (self, element):
        return self.left <= element.left and self.right >= element.right
    
class Bridge (Enemy):
    color = [(20, 143, 147), (40, 223, 227)] # Dark yellow
    area = [200, 1000]

    def __init__(self, position):
        super().__init__("Bridge", position, width=100)