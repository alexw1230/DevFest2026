# =================== Quest System ===================
class QuestManager:
    def __init__(self):
        self.main_quest = "Survive the Simulation"
        self.side_quests = [
            "Scan a high-mana entity",
            "Identify 3 friendly NPCs",
            "Maintain distance from threats",
            "Find the hidden merchant",
            "Analyze biological signatures"
        ]
        self.current_side_quest = random.choice(self.side_quests)
        self.quest_color = (0, 255, 255) # Gold/Yellow

    def refresh_side_quest(self):
        self.current_side_quest = random.choice(self.side_quests)

quest_sys = QuestManager()


def draw_quest_ui(frame, quest_manager):
    """Draws a semi-transparent Quest Log in the top right."""
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    # UI Box Dimensions
    box_w, box_h = 300, 120
    tx, ty = w - box_w - 20, 20
    
    # Draw Background Semi-transparent Box
    cv2.rectangle(overlay, (tx, ty), (tx + box_w, ty + box_h), (40, 40, 40), -1)
    cv2.rectangle(overlay, (tx, ty), (tx + box_w, ty + box_h), (200, 200, 200), 2)
    
    # Apply transparency
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Text
    cv2.putText(frame, "QUEST LOG", (tx + 10, ty + 25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(frame, (tx + 10, ty + 35), (tx + box_w - 10, ty + 35), (150, 150, 150), 1)
    
    # Main Quest
    cv2.putText(frame, f"Main: {quest_manager.main_quest}", (tx + 10, ty + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    
    # Side Quest
    cv2.putText(frame, f"Side: {quest_manager.current_side_quest}", (tx + 10, ty + 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)


