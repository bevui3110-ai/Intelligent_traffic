import math

class EuclideanDistTracker:
    def __init__(self, dist_thresh=90, max_disappeared=5):
        # Lưu vị trí trung tâm, class_index và số frame bị biến mất: {id: {'pos': (cx, cy), 'index': class_index, 'disappeared': 0}}
        self.center_points = {}
        # Đếm tổng số ID đã cấp
        self.id_count = 0
        self.dist_thresh = dist_thresh
        self.max_disappeared = max_disappeared


    def update(self, objects_rect):
        objects_bbs_ids = []
        matched_ids = set()

        for rect in objects_rect:
            x, y, w, h, index = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            min_dist = float('inf')
            closest_id = None

            # Tìm đối tượng gần nhất cùng loại xe chưa được gán trong frame này
            for obj_id, data in self.center_points.items():
                if obj_id in matched_ids:
                    continue
                pt_x, pt_y = data['pos']
                pt_index = data['index']

                if pt_index == index:
                    dist = math.hypot(cx - pt_x, cy - pt_y)
                    adaptive_thresh = max(self.dist_thresh, int(max(w, h) * 0.75))
                    if dist < min_dist and dist < adaptive_thresh:
                        min_dist = dist
                        closest_id = obj_id

            if closest_id is not None:
                self.center_points[closest_id]['pos'] = (cx, cy)
                self.center_points[closest_id]['disappeared'] = 0
                matched_ids.add(closest_id)
                objects_bbs_ids.append([x, y, w, h, closest_id, index])
                same_object_detected = True

            # Nếu là đối tượng mới xuất hiện hoàn toàn
            if not same_object_detected:
                self.center_points[self.id_count] = {'pos': (cx, cy), 'index': index, 'disappeared': 0}
                matched_ids.add(self.id_count)
                objects_bbs_ids.append([x, y, w, h, self.id_count, index])
                self.id_count += 1

        # Tăng số frame biến mất cho các ID không tìm thấy trong frame hiện tại & xóa ID quá cũ
        unmatched_ids = set(self.center_points.keys()) - matched_ids
        for obj_id in list(unmatched_ids):
            self.center_points[obj_id]['disappeared'] += 1
            if self.center_points[obj_id]['disappeared'] > self.max_disappeared:
                del self.center_points[obj_id]

        return objects_bbs_ids
