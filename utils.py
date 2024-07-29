import cv2 as cv
import numpy as np
import mediapipe as mp
import tempfile
import math
import time


def triangle_to_rectangle(a, b, c):
    # Calculam punctul d, considerand triunghiul abc, care reprezinta coordonatele piciorului
    # Deoarece c este calcaiul, trebuie sa aflam distanta de la c la unul dintre pct a si b
    # Si sa adaugam distanta la celalalt pct => pct d
    x = b[0] + (a[0] - c[0])
    y = b[1] + (a[1] - c[1])
    return [x, y]


def line_to_rectangle(a, b, length):
    try:
        slope = (b[1] - a[1]) / (b[0] - a[0])
        dy = math.sqrt(length ** 2 / (slope ** 2 + 1))
        dx = -slope * dy
        c = [a[0] + dx, a[1] + dy]
        d = [a[0] - dx, a[1] - dy]
        e = [b[0] + dx, b[1] + dy]
        f = [b[0] - dx, b[1] - dy]
    except:
        c = [a[0] + length, a[1] + length]
        d = [a[0] - length, a[1] - length]
        e = [b[0] + length, b[1] + length]
        f = [b[0] - length, b[1] - length]

    return [c, d, f, e]


def extend_points(a, b, margin):
    t0 = 0.5 * (1.0 - margin)
    t1 = 0.5 * (1.0 + margin)
    x1 = round(a[0] + (b[0] - a[0]) * t0)
    y1 = round(a[1] + (b[1] - a[1]) * t0)
    x2 = round(a[0] + (b[0] - a[0]) * t1)
    y2 = round(a[1] + (b[1] - a[1]) * t1)
    return x1, y1, x2, y2


def add_margins(pts, margin):
    pts[0][0], pts[0][1], pts[1][0], pts[1][1] = extend_points(pts[0], pts[1], margin)
    pts[3][0], pts[3][1], pts[2][0], pts[2][1] = extend_points(pts[3], pts[2], margin)
    pts[0][0], pts[0][1], pts[3][0], pts[3][1] = extend_points(pts[0], pts[3], margin)
    pts[1][0], pts[1][1], pts[2][0], pts[2][1] = extend_points(pts[1], pts[2], margin)
    return pts


def sort_pts(points, verify=False):
    sorted_pts = np.zeros((4, 2), dtype="float32")
    s = np.sum(points, axis=1)
    sorted_pts[0] = points[np.argmin(s)]
    sorted_pts[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    sorted_pts[1] = points[np.argmin(diff)]
    sorted_pts[3] = points[np.argmax(diff)]

    if verify:
        if np.array_equal(sorted_pts[2], points[2]) or np.array_equal(sorted_pts[2], points[3]):
            if np.array_equal(sorted_pts[3], points[2]) or np.array_equal(sorted_pts[3], points[3]):
                return sorted_pts

    return np.array([points[0], points[1], points[2], points[3]])


def face_fronting_foot(a, b, c, min):
    dist = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    dist2 = math.sqrt((c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2)
    if dist < min or dist2 < min:
        return True
    else:
        return False


def foot_pos(png, idx_up, idx_heel):
    if idx_up == 0 or idx_up == 1:
        if idx_heel == 0:
            png = cv.rotate(png, cv.ROTATE_90_CLOCKWISE)
        elif idx_heel == 1:
            png = cv.rotate(png, cv.ROTATE_90_CLOCKWISE)
            png = cv.flip(png, 1)
        elif idx_heel == 2:
            png = cv.flip(png, 1)
    else: # idx_up == 2 sau 3
        if idx_heel == 0:
            png = cv.rotate(png, cv.ROTATE_180)
            png = cv.flip(png, 1)
        elif idx_heel == 1:
            png = cv.rotate(png, cv.ROTATE_180)
        elif idx_heel == 2:
            png = cv.rotate(png, cv.ROTATE_90_COUNTERCLOCKWISE)
        else:
            png = cv.rotate(png, cv.ROTATE_90_COUNTERCLOCKWISE)
            png = cv.flip(png, 1)
    return png


def rotate_image(img, angle):
    h, w = img.shape[:2]
    if h > w:
        border = h // 2
    else:
        border = w // 2
    temp_img = cv.copyMakeBorder(img, top=border, bottom=border, left=border, right=border,
                                 borderType=cv.BORDER_CONSTANT,
                                 value=[0, 0, 0, 0])

    size_reverse = np.array(temp_img.shape[1::-1])  # swap x with y
    M = cv.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.0)
    rotated = cv.warpAffine(temp_img, M, size_reverse)

    return rotated, border


def add_head(frame_shape, png, bg, landmarks, model):
    # Calculam centrul fetei
    x = [landmarks[0].x * frame_shape[1], landmarks[7].x * frame_shape[1], landmarks[8].x * frame_shape[1]]
    y = [landmarks[0].y * frame_shape[0], landmarks[7].y * frame_shape[0], landmarks[8].y * frame_shape[0]]
    center_face = (round(sum(x) / 3), round(sum(y) / 3))

    # lungimea dintre urechi
    len_main = round(np.sqrt((landmarks[8].y * frame_shape[0] - landmarks[7].y * frame_shape[0]) ** 2 +
                             (landmarks[8].x * frame_shape[1] - landmarks[7].x * frame_shape[1]) ** 2))
    # lungimea dintre urechia din dreapta si nasul
    len_right = round(np.sqrt((landmarks[0].y * frame_shape[0] - landmarks[7].y * frame_shape[0]) ** 2 +
                              (landmarks[0].x * frame_shape[1] - landmarks[7].x * frame_shape[1]) ** 2))
    # lungimea dintre urechia din stanga si nasul
    len_left = round(np.sqrt((landmarks[0].y * frame_shape[0] - landmarks[8].y * frame_shape[0]) ** 2 +
                             (landmarks[0].x * frame_shape[1] - landmarks[8].x * frame_shape[1]) ** 2))

    flip_img = False
    if len_left < len_right or y[0] < y[2]:
        flip_img = True
        png = cv.flip(png, 1)

    # Aproximam latimea capului in functie de datele anterioare
    if len_main > len_right and len_main > len_left:
        head_len = len_main * 2.0
    elif len_right > len_left:
        head_len = len_right * 2.25
    else:
        head_len = len_left * 2.25

    if model == "models/model2/":
        head_len = head_len * 1.25

    # Redimensionare png_head in functie de dimensiunile din imagine
    w_head = png.shape[1]
    diff = float(head_len / w_head)
    png = cv.resize(png, (int(png.shape[1] * diff), int(png.shape[0] * diff)))

    h, w = png.shape[:2]  # dimensiuni noi alea png head

    # In cazul in care capul este sub umeri, persoana sta in maini etc.
    if landmarks[12].y < landmarks[0].y or landmarks[11].y < landmarks[0].y:
        png = cv.flip(png, 0)
        add_to_end = round(2 * h / 3)
        add_to_start = round(h / 3)
    else:
        add_to_start = round(2 * h / 3)
        add_to_end = round(h / 3)

    # Incercam sa folosim head_tilt pentru a modifica imaginea in functie de pozitia capului
    try:
        head_tilt = -1 * round(np.rad2deg(np.arctan((landmarks[8].y * frame_shape[0] - landmarks[7].y * frame_shape[0])
                                                    / (landmarks[8].x * frame_shape[1] - landmarks[7].x * frame_shape[1]))))
        png, border_head = rotate_image(png, head_tilt)
    except:
        border_head = 0
        pass

    # Punctele necesare x-width, y-height
    start_x = center_face[0] - w // 2 - round(border_head)
    start_y = center_face[1] - add_to_start - round(border_head)
    end_x = center_face[0] + round(w / 2 + 0.1) + round(border_head)
    end_y = center_face[1] + add_to_end + round(border_head)

    # Verificam sa nu fim prea aproape de margini / persoana a iesit din cadru
    fig_center = [(start_x + end_x) / 2, (start_y + end_y) / 2]
    if fig_center[0] > 20 and fig_center[1] > 20:
        # Se separa culorile de canalul alpha pentru toate pozele png
        head_alpha_ch = png[:, :, 3] / 255  # 0-255 --> 0.0-1.0
        head_overlay = png[:, :, :3]
        head_alpha_mask = np.dstack((head_alpha_ch, head_alpha_ch, head_alpha_ch))

        # Modificari pentru depasirea marginilor
        if start_x < 0:
            height, width = head_overlay.shape[:2]
            head_overlay = head_overlay[0:height, abs(start_x):width]
            head_alpha_mask = head_alpha_mask[0:height, abs(start_x):width]
            start_x = 0
        if start_y < 0:
            height, width = head_overlay.shape[:2]
            head_overlay = head_overlay[abs(start_y):height, 0:width]
            head_alpha_mask = head_alpha_mask[abs(start_y):height, 0:width]
            start_y = 0
        if end_x > frame_shape[1]:
            height, width = head_overlay.shape[:2]
            head_overlay = head_overlay[0:height, 0:width - (end_x - frame_shape[1])]
            head_alpha_mask = head_alpha_mask[0:height, 0:width - (end_x - frame_shape[1])]
            end_x = frame_shape[1]
        if end_y > frame_shape[0]:
            height, width = head_overlay.shape[:2]
            head_overlay = head_overlay[0:height - (end_y - frame_shape[0]), 0:width]
            head_alpha_mask = head_alpha_mask[0:height - (end_y - frame_shape[0]), 0:width]
            end_y = frame_shape[0]

        # Adaugare png la imaginea noastra
        img_subsection = bg[start_y:end_y, start_x:end_x]

        # Verificam daca imaginea png are dimensiunile necesare pentru a fi afisata
        sw = True
        if img_subsection.shape[0] == 0 or img_subsection.shape[1] == 0:
            sw = False
        if head_alpha_mask.shape[0] == 0 or head_alpha_mask.shape[1] == 0:
            sw = False

        if sw:
            composite = img_subsection * (1 - head_alpha_mask) + head_overlay * head_alpha_mask
            bg[start_y:end_y, start_x:end_x] = composite

    return bg, head_len, flip_img


def add_png(cam, png, corners):
    h, w = png.shape[:2]

    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts2 = np.float32(corners)

    # Get the transformation matrix and use it to get the warped image of the subject
    M = cv.getPerspectiveTransform(pts1, pts2)
    warped_img = cv.warpPerspective(png, M, (cam.shape[1], cam.shape[0]))

    alpha_ch = warped_img[:, :, 3] / 255  # 0-255 --> 0.0-1.0
    overlay = warped_img[:, :, :3]
    alpha_mask = np.dstack((alpha_ch, alpha_ch, alpha_ch))

    cam[:] = cam * (1 - alpha_mask) + overlay * alpha_mask
    return cam


def process_frame(img_shape, landmarks, model='models/model1/', bg=None):
    # Citim imaginile png pentru modificari
    # Trebuie citite de fiecare data pentru a nu isi pierde calitatea sau pentru a nu fi decupate
    png_head = cv.imread(model + "head.png", cv.IMREAD_UNCHANGED)  # merge
    png_torso = cv.imread(model + "torso.png", cv.IMREAD_UNCHANGED)
    png_lat_torso = cv.imread(model + "lat_torso.png", cv.IMREAD_UNCHANGED)
    png_upper_arm = cv.imread(model + "upper_arm.png", cv.IMREAD_UNCHANGED)
    png_lower_arm = cv.imread(model + "lower_arm.png", cv.IMREAD_UNCHANGED)
    png_upper_leg = cv.imread(model + "upper_leg.png", cv.IMREAD_UNCHANGED)
    png_lower_leg = cv.imread(model + "lower_leg.png", cv.IMREAD_UNCHANGED)
    png_foot = cv.imread(model + "foot.png", cv.IMREAD_UNCHANGED)
    png_front_foot = cv.imread(model + "front_foot.png", cv.IMREAD_UNCHANGED)

    # HEAD -----------------------------------------------------------------------------------------------------
    bg, head_len, flip_img = add_head(img_shape, png_head, bg, landmarks, model)

    # FOOT -----------------------------------------------------------------------------------------------------
    # RIGHT
    a = [landmarks[27].x * img_shape[1], landmarks[27].y * img_shape[0]]
    b = [landmarks[31].x * img_shape[1], landmarks[31].y * img_shape[0]]
    c = [landmarks[29].x * img_shape[1], landmarks[29].y * img_shape[0]]
    d = triangle_to_rectangle(a, b, c)
    points = [a, d, b, c]

    if face_fronting_foot(c, a, b, head_len * 0.2):
        half_mark = [round((b[0] + c[0]) / 2) , round((b[1] + c[1]) / 2)]
        _, _, x2, y2 = extend_points(a, half_mark, margin=2.5)
        half_mark = [x2, y2]

        points = line_to_rectangle(a, half_mark, head_len * 0.2)
        sorted_pts = add_margins(points, margin=1.2)
        sorted_pts = sort_pts(sorted_pts, verify=True)
        bg = add_png(bg, png_front_foot, sorted_pts)
    else:
        points = add_margins(points, margin=1.4)
        sort_pts(points, verify=True)
        idx1 = points.index(a)
        idx2 = points.index(c)
        png_foot_copy = png_foot.copy()
        foot_pos(png_foot_copy, idx1, idx2)

        bg = add_png(bg, png_foot, points)

    # LEFT
    a = [landmarks[28].x * img_shape[1], landmarks[28].y * img_shape[0]]
    b = [landmarks[32].x * img_shape[1], landmarks[32].y * img_shape[0]]
    c = [landmarks[30].x * img_shape[1], landmarks[30].y * img_shape[0]]
    d = triangle_to_rectangle(a, b, c)
    points = [a, d, b, c]

    if face_fronting_foot(c, a, b, head_len * 0.2):
        half_mark = [round((b[0] + c[0]) / 2), round((b[1] + c[1]) / 2)]
        _, _, x2, y2 = extend_points(a, half_mark, margin=2.5)
        half_mark = [x2, y2]

        points = line_to_rectangle(a, half_mark, head_len * 0.2)
        sorted_pts = add_margins(points, margin=1.2)
        sorted_pts = sort_pts(sorted_pts, verify=True)
        bg = add_png(bg, png_front_foot, sorted_pts)
    else:
        points = add_margins(points, margin=1.4)
        sort_pts(points, verify=True)
        idx1 = points.index(a)
        idx2 = points.index(c)
        png_foot_copy = png_foot.copy()
        foot_pos(png_foot_copy, idx1, idx2)

        bg = add_png(bg, png_foot, points)

    # LOWER LEGS -----------------------------------------------------------------------------------------------
    # RIGHT
    a = (landmarks[25].x * img_shape[1], landmarks[25].y * img_shape[0])
    b = (landmarks[27].x * img_shape[1], landmarks[27].y * img_shape[0])
    length = 0.2 * head_len
    points = line_to_rectangle(a, b, length)

    sorted_pts = add_margins(points, margin=1.1)
    sorted_pts = sort_pts(sorted_pts)

    bg = add_png(bg, png_lower_leg, sorted_pts)

    # LEFT
    a = (landmarks[26].x * img_shape[1], landmarks[26].y * img_shape[0])
    b = (landmarks[28].x * img_shape[1], landmarks[28].y * img_shape[0])
    length = 0.2 * head_len
    points = line_to_rectangle(a, b, length)

    sorted_pts = add_margins(points, margin=1.1)
    sorted_pts = sort_pts(sorted_pts)

    # Trebuie oglindita imaginea
    temp_img = cv.flip(png_lower_leg, 1)
    bg = add_png(bg, temp_img, sorted_pts)

    # UPPERR LEGS -----------------------------------------------------------------------------------------------
    # RIGHT
    a = (landmarks[23].x * img_shape[1], landmarks[23].y * img_shape[0])
    b = (landmarks[25].x * img_shape[1], landmarks[25].y * img_shape[0])
    length = 0.25 * head_len
    points = line_to_rectangle(a, b, length)

    sorted_pts = add_margins(points, margin=1.1)
    sorted_pts = sort_pts(sorted_pts)

    bg = add_png(bg, png_upper_leg, sorted_pts)

    # LEFT
    a = (landmarks[24].x * img_shape[1], landmarks[24].y * img_shape[0])
    b = (landmarks[26].x * img_shape[1], landmarks[26].y * img_shape[0])
    length = 0.25 * head_len
    points = line_to_rectangle(a, b, length)

    sorted_pts = add_margins(points, margin=1.1)
    sorted_pts = sort_pts(sorted_pts)

    # Trebuie oglindita imaginea
    temp_img = cv.flip(png_upper_leg, 1)
    bg = add_png(bg, temp_img, sorted_pts)

    # TORSO ----------------------------------------------------------------------------------------------------
    # Extragem punctele pentru trunchiul corpului
    points = [[landmarks[12].x * img_shape[1], landmarks[12].y * img_shape[0]],
              [landmarks[11].x * img_shape[1], landmarks[11].y * img_shape[0]],
              [landmarks[23].x * img_shape[1], landmarks[23].y * img_shape[0]],
              [landmarks[24].x * img_shape[1], landmarks[24].y * img_shape[0]]]
    shoulder = math.sqrt((points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2)

    if shoulder < (head_len // 2):
        x1, y1, x2, y2 = extend_points(points[0], points[1], margin=2.5)
        points[0] = [x1, y1]
        points[1] = [x2, y2]
        x1, y1, x2, y2 = extend_points(points[2], points[3], margin=1.3)
        points[2] = [x1, y1]
        points[3] = [x2, y2]
        sorted_pts = add_margins(points, margin=1.2)
        sorted_pts = sort_pts(sorted_pts, verify=True)

        if np.array_equal(sorted_pts[0], sorted_pts[1]):
            sorted_pts[0][0] = sorted_pts[0][0] - head_len//2.5
            sorted_pts[1][0] = sorted_pts[1][0] + head_len//2.5
            sorted_pts = sort_pts(sorted_pts, verify=True)
        if np.array_equal(sorted_pts[2], sorted_pts[3]):
            sorted_pts[2][0] = sorted_pts[2][0] + head_len//3.5
            sorted_pts[3][0] = sorted_pts[3][0] - head_len//3.5
            sorted_pts = sort_pts(sorted_pts, verify=True)



        if flip_img:
            png_lat_torso = cv.flip(png_lat_torso, 1)
        bg = add_png(bg, png_lat_torso, sorted_pts)

    else:
        x1, y1, x2, y2 = extend_points(points[2], points[3], margin=1.3)
        points[2] = [x1, y1]
        points[3] = [x2, y2]
        sorted_pts = add_margins(points, margin=1.3)
        sorted_pts = sort_pts(sorted_pts, verify=True)

        if flip_img:
            png_torso = cv.flip(png_torso, 1)
        bg = add_png(bg, png_torso, sorted_pts)

    # UPPER ARMS -----------------------------------------------------------------------------------------------
    # RIGHT
    a = (landmarks[11].x * img_shape[1], landmarks[11].y * img_shape[0])
    b = (landmarks[13].x * img_shape[1], landmarks[13].y * img_shape[0])
    length = 0.12 * head_len
    points = line_to_rectangle(a, b, length)

    sorted_pts = add_margins(points, margin=1.3)
    sorted_pts = sort_pts(sorted_pts)

    bg = add_png(bg, png_upper_arm, sorted_pts)

    # LEFT
    a = (landmarks[12].x * img_shape[1], landmarks[12].y * img_shape[0])
    b = (landmarks[14].x * img_shape[1], landmarks[14].y * img_shape[0])
    length = 0.12 * head_len
    points = line_to_rectangle(a, b, length)

    sorted_pts = add_margins(points, margin=1.3)
    sorted_pts = sort_pts(sorted_pts)

    # Trebuie oglindita imaginea
    temp_img = cv.flip(png_upper_arm, 1)
    bg = add_png(bg, temp_img, sorted_pts)

    # LOWER ARMS -----------------------------------------------------------------------------------------------
    # RIGHT
    a = (landmarks[13].x * img_shape[1], landmarks[13].y * img_shape[0])
    b = (landmarks[19].x * img_shape[1], landmarks[19].y * img_shape[0])
    length = 0.15 * head_len
    points = line_to_rectangle(a, b, length)

    sorted_pts = add_margins(points, margin=1.1)
    sorted_pts = sort_pts(sorted_pts)

    bg = add_png(bg, png_lower_arm, sorted_pts)

    # LEFT
    a = (landmarks[14].x * img_shape[1], landmarks[14].y * img_shape[0])
    b = (landmarks[20].x * img_shape[1], landmarks[20].y * img_shape[0])
    length = 0.15 * head_len
    points = line_to_rectangle(a, b, length)

    sorted_pts = add_margins(points, margin=1.1)
    sorted_pts = sort_pts(sorted_pts)

    # Trebuie oglindita imaginea
    temp_img = cv.flip(png_lower_arm, 1)
    bg = add_png(bg, temp_img, sorted_pts)

    return bg


def animate_frame(frame, model='model1', bg=None):
    # Path pt model
    model = "models/" + model + "/"

    # Create bg image for the animation
    if bg is None:
        bg = np.zeros([frame.shape[0], frame.shape[1], 3], dtype=np.uint8)
        bg.fill(255)
    elif type(bg) == str:
        bg = cv.imread('app/static/backgrounds/'+bg+'.jpg')
        bg = cv.resize(bg, (frame.shape[1], frame.shape[0]))
    else:
        bg = cv.resize(bg, (frame.shape[1], frame.shape[0]))

    # Configurare elemente mediapipe
    mp_pose = mp.solutions.pose

    # Initializare model mediapipe
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:

        # Schimbare din BGR in RGB
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detectare folosind mediapipe
        results = pose.process(image)

        # Schimbare inapoi in BGR pentru modificari
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Dimensiuni cadru/imagine
        img_shape = image.shape

        # Extragere landmarks pentru coordonate
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass

        # Verifica daca s-a gasit persoana din cadru si s-au extras landmarks
        if 'landmarks' in locals():
            result_img = process_frame(img_shape, landmarks, model, bg)
        else:
            result_img = bg

    return result_img


def animate_video(video, model='model1', bg=None):
    # Path pt model
    model = "models/" + model + "/"

    # Configurare detectie mediapipe
    mp_pose = mp.solutions.pose

    # Se foloseste video
    cap = cv.VideoCapture(video)

    # Inregistrare camera
    fps = cap.get(cv.CAP_PROP_FPS)
    cam_w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    cam_h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    # Video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    videoWriter = cv.VideoWriter(temp_output.name, fourcc, fps, (int(cam_w), int(cam_h)))

    if not videoWriter.isOpened():
        print("Error: VideoWriter failed to open.")
        return None

    # Initializare model mediapipe
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        while cap.isOpened():
            # start_time = time.time()
            ret, frame = cap.read()

            if not ret:
                break

            # Create bg image for the animation
            if bg is None:
                bg = np.zeros([frame.shape[0], frame.shape[1], 3], dtype=np.uint8)
                bg.fill(255)
            elif type(bg) == str:
                bg = cv.imread('app/static/backgrounds/' + bg + '.jpg')
                bg = cv.resize(bg, (frame.shape[1], frame.shape[0]))
            else:
                bg = cv.resize(bg, (frame.shape[1], frame.shape[0]))

            # Schimbare din BGR in RGB
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Detectare folosind mediapipe
            results = pose.process(image)

            # Schimbare inapoi in BGR pentru modificari
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            # Dimensiuni cadru/imagine
            img_shape = image.shape

            # Extragere landmarks pentru coordonate
            try:
                landmarks = results.pose_landmarks.landmark
            except:
                pass

            # Verifica daca s-a gasit persoana din cadru si s-au extras landmarks
            if 'landmarks' in locals():
                processed_frame = process_frame(img_shape, landmarks, model, bg.copy())
                videoWriter.write(processed_frame)
            else:
                processed_frame = bg
                videoWriter.write(processed_frame)

    cap.release()
    videoWriter.release()

    # Citim video din memoria temp
    with open(temp_output.name, 'rb') as f:
        processed_video_data = f.read()

    return processed_video_data

