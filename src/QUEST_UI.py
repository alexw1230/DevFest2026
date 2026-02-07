import cv2
import numpy as np
from ultralytics import YOLO
import random

# =================== Models ===================
det_model = YOLO("models/yolov8n.pt")        # detection + tracking
pose_model = YOLO("models/yolov8n-pose.pt")  # optional, for pose info

cap = cv2.VideoCapture(0)

# =================== Constants ===================
KNOWN_HEIGHT = 1.7  # meters (average person)
FOCAL_LENGTH = 700  # tune based on your camera
MAX_HP = 100
MAX_MANA = 100
THREAT_THRESHOLD = 120  # size_factor threshold for threat
PROXIMITY_THRESHOLD = 50  # pixels, for matching old track

# =================== Persistent attributes ===================
# track_id -> {'hp': HP, 'mana': Mana, 'bbox': (x1,y1,x2,y2)}
person_attributes = {}

# =================== Helper functions ===================
def estimate_distance(bbox_height_px):
    if bbox_height_px == 0:
        return None
    return (KNOWN_HEIGHT * FOCAL_LENGTH) / bbox_height_px

def dominant_color(img):
    img = img.reshape((-1, 3))
    img = np.float32(img)
    _, _, centers = cv2.kmeans(
        img, 1, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )
    return tuple(map(int, centers[0]))

def color_to_mana(rgb):
    r, g, b = rgb
    brightness = (0.299*r + 0.587*g + 0.114*b)
    mana = int((brightness / 255) * MAX_MANA)
    return mana

def size_to_hp(bbox_height, distance):
    if distance is None or bbox_height == 0:
        return 0
    size_factor = bbox_height / distance
    max_size_factor = 300  # adjust for camera/scene
    hp = int(min(size_factor / max_size_factor * MAX_HP, MAX_HP))
    return hp

def get_persistent_attributes(track_id, bbox, attributes_dict, threshold=PROXIMITY_THRESHOLD):
    """
    Returns existing HP/Mana if track_id exists,
    or finds the closest previous bbox within threshold to reuse attributes.
    """
    if track_id in attributes_dict:
        return (attributes_dict[track_id]['hp'], attributes_dict[track_id]['mana']), track_id

    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2

    # Check for previous bbox within threshold
    for old_id, data in attributes_dict.items():
        ox1, oy1, ox2, oy2 = data['bbox']
        ox = (ox1 + ox2) // 2
        oy = (oy1 + oy2) // 2
        distance = ((cx - ox)**2 + (cy - oy)**2)**0.5
        if distance < threshold:
            return (data['hp'], data['mana']), old_id

    return None, track_id

def draw_health_bars(frame, x, y, hp, mana, max_hp=MAX_HP, max_mana=MAX_MANA):
    """Draw RPG-style health and mana bars."""
    bar_width = 150
    bar_height = 20
    spacing = 5
    outline_thickness = 2
    
    # Background panel
    panel_height = bar_height * 2 + spacing * 3
    panel_width = bar_width + 30
    cv2.rectangle(frame, (x - 10, y - 10), (x + panel_width, y + panel_height), (20, 20, 20), -1)
    cv2.rectangle(frame, (x - 10, y - 10), (x + panel_width, y + panel_height), (100, 100, 100), outline_thickness)
    
    # HP Bar
    hp_ratio = max(0, min(hp / max_hp, 1.0))
    hp_bar_width = int(bar_width * hp_ratio)
    
    # HP background (dark red)
    cv2.rectangle(frame, (x + 5, y + 5), (x + bar_width + 5, y + bar_height + 5), (50, 50, 100), -1)
    # HP fill (bright red)
    cv2.rectangle(frame, (x + 5, y + 5), (x + 5 + hp_bar_width, y + bar_height + 5), (0, 0, 255), -1)
    # HP border
    cv2.rectangle(frame, (x + 5, y + 5), (x + bar_width + 5, y + bar_height + 5), (150, 150, 150), outline_thickness)
    
    # HP text
    cv2.putText(frame, f"HP {hp}/{max_hp}", (x + 10, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Mana Bar
    mana_ratio = max(0, min(mana / max_mana, 1.0))
    mana_bar_width = int(bar_width * mana_ratio)
    
    # Mana background (dark blue)
    cv2.rectangle(frame, (x + 5, y + bar_height + spacing + 5), 
                  (x + bar_width + 5, y + bar_height * 2 + spacing + 5), (100, 50, 50), -1)
    # Mana fill (bright blue)
    cv2.rectangle(frame, (x + 5, y + bar_height + spacing + 5), 
                  (x + 5 + mana_bar_width, y + bar_height * 2 + spacing + 5), (255, 0, 0), -1)
    # Mana border
    cv2.rectangle(frame, (x + 5, y + bar_height + spacing + 5), 
                  (x + bar_width + 5, y + bar_height * 2 + spacing + 5), (150, 150, 150), outline_thickness)
    
    # Mana text
    cv2.putText(frame, f"Mana {mana}/{max_mana}", (x + 10, y + bar_height * 2 + spacing + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

# =================== Fullscreen setup ===================
cv2.namedWindow("YOLOv8 RPG View", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLOv8 RPG View", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)






#=================== Make QuestLine================

Main_quests = ["Get food", "win Hackathon" , "etc"]
# at program start, basic print out the main questline tag
#and type out the main questline, should be randomly picked from array 
#will probably get from chat at some point 

Side_quests = ["have sex", "fuck u motha", "eat ass", 
               "pushing it, pushing me", "vicariously IIIII", 
               "live while the whole world dies"
               
               
               
               
               
               ] # also from chat at some point 
'''
#=======code i wanna push, needs procedding code to function ==========
sidequests = random.sample(Side_quests, k = 3)



B_leftx, B_lefty = 50, 50
B_leftw, B_lefth = 400, 300


cv2.rectangle(frame, (B_leftx, B_lefty), (B_leftx+B_leftw, B_lefty+B_lefth), (0, 0, 255), 50, 2)

cv2.putText(frame, "QUEST LOG", (B_leftx+10, B_lefty+10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 20.2, 3, 3)


cv2.putText(frame, "Main Quest", (B_leftx + 20, B_lefty + 20), cv2.FONT_HERSHEY_TRIPLEX, 20.2, (50,50,50), 4)
cv2.putText(frame, Main_quests, (B_leftx + 20, B_lefty + 80), cv2.FONT_HERSHEY_TRIPLEX, 20.2, (50,50,50), 4)

cv2.putText(
    frame,
    "Side Quests:",
    (B_leftx + 20, B_lefty + 160),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

y_offset = 190
for quest in sidequests:
    cv2.putText(frame, f"-{quest}", (B_leftx + 20, B_lefty + y_offset), cv2.FONT_HERSHEY_PLAIN, 20.2, (50,50,50), 4 )
    y_offset += 30 


#=======END OF: code i wanna push =======

'''




sidequests_current = random.sample(Side_quests, k = 3)


main_quest_current = random.choice(Main_quests)


##=======code i wanna push, needs procedding code to function ==========
def Questline_UI(frame, sidequests, main_quest):
    
    
    

    

# make height and width from frame
    h, w = frame.shape[:2]

    panel_w = 420
    panel_h = 260
    margin = 20

    B_leftx = margin
    B_lefty = h - panel_h - margin

#B_leftx, B_lefty = x1 + 50, y1 + 50
#B_leftw, B_lefth = x2 + 400, y2 + 300

    overlay = frame.copy()
    cv2.rectangle(
    overlay,
    (B_leftx, B_lefty),
    (B_leftx + panel_w, B_lefty + panel_h),
    (30, 30, 30),
    -1
)

    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)




#title 
    cv2.putText(frame, "QUEST LOG", (B_leftx+20, B_lefty+35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255,255,255), 1)

#main quest
    cv2.putText(frame, "Main Quest", (B_leftx + 20, B_lefty + 75), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, main_quest, (B_leftx + 20, B_lefty + 100), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (255,255,255), 2)

    y = 145
    cv2.putText(
    frame,
    "Side Quests:",
    (B_leftx + 20, B_lefty + y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
    )

    y_offset = y + 30
    for quest in sidequests:
        cv2.putText(frame, f"-{quest}", (B_leftx + 20, B_lefty + y_offset), cv2.FONT_HERSHEY_PLAIN, 0.9, (200,200,200), 1)
        y_offset += 30 
    return frame 



#=======END OF: code i wanna push =======


# =================== Main loop ===================
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # Run detection + tracking
    results = det_model.track(
        frame,
        persist=True,
        classes=[0],  # only people
        conf=0.4,
        tracker="bytetrack.yaml"
    )

    for r in results:
        if r.boxes.id is None:
            continue

        for box, track_id in zip(r.boxes, r.boxes.id):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_height = y2 - y1
            distance = estimate_distance(bbox_height)
            if distance is None:
                continue

            # =================== Persistent HP/Mana ===================
            attr, matched_id = get_persistent_attributes(track_id, (x1, y1, x2, y2), person_attributes)
            if attr:
                hp, mana = attr
            else:
                # First time seeing this person
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue
                h = person_crop.shape[0]
                upper = person_crop[:h//2, :]
                upper_color = dominant_color(upper)

                hp = size_to_hp(bbox_height, distance)
                mana = color_to_mana(upper_color)

            # Update dictionary with current bbox
            person_attributes[matched_id] = {'hp': hp, 'mana': mana, 'bbox': (x1, y1, x2, y2)}

            # =================== Threat logic ===================
            size_factor = bbox_height / distance
            threat = size_factor > THREAT_THRESHOLD
            box_color = (0, 0, 255) if threat else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Draw RPG-style health and mana bars above the person
            bar_x = (x1 + x2) // 2 - 75  # Center horizontally (bar_width/2 = 75)
            bar_y = max(10, y1 - 70)  # Above the person, with safety margin
            draw_health_bars(frame, bar_x, bar_y, hp, mana)

    

    
   

    frame = Questline_UI(frame, sidequests_current, main_quest_current)
       
            


  


    cv2.imshow("YOLOv8 RPG View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('r'): # Refresh quests manually
        current_main = random.choice(Main_quests)
        current_sides = random.sample(Side_quests, k=3)

cap.release()
cv2.destroyAllWindows()

print("MAIN:", main_quest)
print("SIDE:", sidequests)
