import cv2
import mediapipe as mp
import numpy as np


class FaceEyeTracker:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh=None
        self.cap = cv2.VideoCapture(0)
        self.thickness = 1
        self.circle_radius = 1
        self.drawing_spec = self.mp_drawing.DrawingSpec(self.thickness, self.circle_radius )
        self.eye_coordinates = {
        'left_eye': {
        'x': 0,
        'y': 0},
        'right_eye': {
        'x': 0,
        'y': 0 }
        }
    


   
    
    def Define_draw_landmarks(self,image,face_landmarks):
        self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    
        self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
        self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())
        
    def DefineMesh(self):
       
        face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return face_mesh
          

   
    def process_image_with_face_mesh(self,image,face_mesh):
        result = face_mesh.process(image)
        if result:
            image.flags.writeable = result
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, result
    
    def CalculateEyeCoordinate(self,image,face_landmarks):
        
        left_eye_landmarks = face_landmarks.landmark[468]  
        right_eye_landmarks = face_landmarks.landmark[473]  

        left_eye_cordx = int(left_eye_landmarks.x * image.shape[1])
        self.eye_coordinates['left_eye']['x'] = left_eye_cordx
        left_eye_cordy = int(left_eye_landmarks.y * image.shape[0])
        self.eye_coordinates['left_eye']['y'] = left_eye_cordy
        
        right_eye_cordx = int(right_eye_landmarks.x * image.shape[1])
        self.eye_coordinates['right_eye']['x'] = right_eye_cordx
        right_eye_cordy = int(right_eye_landmarks.y * image.shape[0])
        self.eye_coordinates['right_eye']['y'] = right_eye_cordy
        return left_eye_landmarks,right_eye_landmarks
    
    def calculate_angle(self,point1, point2):
        return (180 / 3.1415) * np.arctan2(point2.y - point1.y, point2.x - point1.x)   
    
    def DrawIrisFocusLine(self,left_eye_landmarks,right_eye_landmarks,image,gaze_angle):
        left_endpoint_x = int(left_eye_landmarks.x * image.shape[1] + 30 * np.cos(gaze_angle))
        left_endpoint_y = int(left_eye_landmarks.y * image.shape[0] + 30 * np.sin(gaze_angle))

        
        cv2.line(image, (int(left_eye_landmarks.x * image.shape[1]), int(left_eye_landmarks.y * image.shape[0])),
                (left_endpoint_x, left_endpoint_y),
                (0, 255, 0), 2)

        
        right_endpoint_x = int(right_eye_landmarks.x * image.shape[1] + 30 * np.cos(gaze_angle))
        right_endpoint_y = int(right_eye_landmarks.y * image.shape[0] + 30 * np.sin(gaze_angle))

     
        cv2.line(image, (int(right_eye_landmarks.x * image.shape[1]), int(right_eye_landmarks.y * image.shape[0])),
                (right_endpoint_x, right_endpoint_y),
                (0, 255, 0), 2)        
    def main(self):
        
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
            
                continue
            face_mesh=self.DefineMesh()
            image,result=self.process_image_with_face_mesh(image,face_mesh)    
            
            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                   
                    
                    self.Define_draw_landmarks(image,face_landmarks)

                    left_eye_landmarks,right_eye_landmarks=self.CalculateEyeCoordinate(image,face_landmarks)
                    gaze_angle = self.calculate_angle(left_eye_landmarks, right_eye_landmarks)
                    self.DrawIrisFocusLine(left_eye_landmarks,right_eye_landmarks,image,gaze_angle)
                    
                  
                   
            cv2.imshow('PsychoTechnic', cv2.flip(image, 1))
            
            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_eye_tracker = FaceEyeTracker()
    face_eye_tracker.main()