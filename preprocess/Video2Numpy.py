import numpy as np # type: ignore
import cv2
import matplotlib.pyplot as plt
import os
def getOpticalFlow(video):
    """Calculate dense optical flow of input video
    Args:
        video: the input video with shape of [frames,height,width,channel]. dtype=np.array
    Returns:
        flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
        flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
    """
    # initialize the list of optical flows
    gray_video = []
    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
        gray_video.append(np.reshape(img,(224,224,1)))

    flows = []
    for i in range(0,len(video)-1):
        # calculate optical flow between each pair of frames
        flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i+1], None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # subtract the mean in order to eliminate the movement of camera
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        # normalize each component in optical flow
        flow[..., 0] = cv2.normalize(flow[..., 0],None,0,255,cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1],None,0,255,cv2.NORM_MINMAX)
        # Add into list 
        flows.append(flow)
        
    # Padding the last frame as empty array
    flows.append(np.zeros((224,224,2)))
      
    return np.array(flows, dtype=np.float32)


def Video2Npy(file_path, resize=(224,224)):
    """Load video and tansfer it into .npy format
    Args:
        file_path: the path of video file
        resize: the target resolution of output video
    Returns:
        frames: gray-scale video
        flows: magnitude video of optical flows 
    """
    # Load video
    cap = cv2.VideoCapture(file_path)
    # Get number of frames
    len_frames = int(cap.get(7))
    # Extract frames from video
    try:
        frames = []
        for i in range(len_frames-1):
            _, frame = cap.read()
            frame = cv2.resize(frame,resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (224,224,3))
            frames.append(frame)   
    except:
        print("Error: ", file_path, len_frames,i)
    finally:
        frames = np.array(frames)
        cap.release()
            
    # Get the optical flow of video
    flows = getOpticalFlow(frames)
    
    result = np.zeros((len(flows),224,224,5))
    result[...,:3] = frames
    result[...,3:] = flows
    
    return result


def FGM_video_results(model, video_path):
    frames = Video2Npy(file_path=video_path)

    predictions = []
    remaining_frames = len(frames) % 64
    for i in range(int(len(frames)/64)):
        preds = model.predict(np.expand_dims(frames[64 * i: 64 * (i + 1)], axis=0))
        
        # Làm tròn kết quả về dạng số thập phân với 2 chữ số sau dấu phẩy
        preds = np.round(preds, 2)
        predictions.extend(preds)
        # In kết quả đã được định dạng
        print(f'Prediction of {64 * i} to {64 * (i + 1) - 1} is {preds}\n')
    # Handling remaining frames

    if remaining_frames > 0:
        # Lặp lại khung hình cuối cùng cho đến khi đạt được 64 khung hình
        pad_size = 64 - remaining_frames
        padding_frames = np.tile(frames[-1], (pad_size, 1, 1, 1))
        
        # Ghép các khung hình còn lại với các khung hình được lặp lại
        padded_frames = np.concatenate((frames[-remaining_frames:], padding_frames), axis=0)
        
        preds = model.predict(np.expand_dims(padded_frames, axis=0))
        
        # Định dạng và in kết quả
        preds = np.round(preds, 2)
        predictions.extend(preds)

        print(f'Prediction of remaining {remaining_frames} frames is {preds}\n')


    # Dữ liệu giả định
    predictions = np.array(predictions)
    # print(predictions)
    # Các lớp
    classes = ['Fight', 'Nonfight']

    # Tổng số khung hình và số khung hình mỗi dự đoán kéo dài
    total_frames = len(frames)
    # print(total_frames)
    frames_per_prediction = [64, 64, remaining_frames]

    # Tính tổng số giây cho mỗi phần của GIF
    total_duration = 5  # Ví dụ: Tổng thời gian chạy GIF là 5 giây
    # durations = [total_duration * (frames / total_frames) for frames in frames_per_prediction]

    # Tạo các khung hình và lưu vào GIF
    filenames = []
    output_dir = './outputs/FGM_barcharts'
    os.makedirs(output_dir, exist_ok=True)
    for i, (prediction, duration) in enumerate(zip(predictions, frames_per_prediction)):
        for j in range(duration):
            filename = f'{output_dir}/barchart_{i:02d}_{j}.png'
            plt.figure(figsize=(3.2, 2.4), dpi=100)
            plt.bar(classes, prediction, color=['blue', 'orange'])
            plt.ylim(0, 1)  # Đặt giới hạn trục y từ 0 đến 1
            plt.title('Prediction')
            
            plt.savefig(filename)
            plt.close()
    print('Process video predict...')
    image_folder = output_dir
    video_name = './outputs/outputs_video/FGM_outputs.avi'

    # Lấy danh sách các tệp ảnh trong thư mục và sắp xếp theo thứ tự
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # Đảm bảo các ảnh được sắp xếp theo thứ tự

    # Lấy kích thước của ảnh đầu tiên để xác định kích thước của video
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Định nghĩa codec và tạo đối tượng VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Chọn codec
    video = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))

    # Ghi từng ảnh vào video
    for image in images:
        img_path = os.path.join(image_folder, image)
        video.write(cv2.imread(img_path))

    # Giải phóng tài nguyên
    video.release()
    cv2.destroyAllWindows()
    
def YOLOv8_video_results(model, video_path):
    results = model.predict(video_path, save=False, imgsz=224, stream = True)  # generator of Results objects
    names = model.names
    # print(names)
    # List of class labels
    class_labels = ['Fight', 'Nonfight']
    output_dir = './outputs/YOLOv8_barcharts'
    os.makedirs(output_dir, exist_ok=True)
    # Lặp qua từng kết quả dự đoán
    for i, result in enumerate(results):
        class_probs = result.probs  # Truy cập vào xác suất của các class
        classname = {
            'Fight':0,
            'NonFight':1
        }
        for ind, prob in enumerate(class_probs):
            name = result.names[ind]  # Lấy tên của class theo chỉ số
            classname[name] = np.round(prob.data.cpu().numpy(), 2) # Chuyển xác suất thành số thực
    #         print(f"Class: {name}, prob: {classname[name]}")
        labels = ['Fight', 'NonFight']
        values = [classname['Fight'], classname['NonFight']]
        filename = f'{output_dir}/barchart_{i:03d}.png'
        plt.figure(figsize=(3.2, 2.4))
        plt.bar(labels, values, color=['blue', 'orange'])
        plt.ylim(0, 1)  # Đặt giới hạn trục y từ 0 đến 1
        plt.title('Prediction of YOLO')
        
        plt.savefig(filename)
        plt.close()
    print('Process video predict...')
    image_folder = output_dir
    video_name = './outputs/outputs_video/YOLOv8_outputs.avi'

    # Lấy danh sách các tệp ảnh trong thư mục và sắp xếp theo thứ tự
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # Đảm bảo các ảnh được sắp xếp theo thứ tự

    # Lấy kích thước của ảnh đầu tiên để xác định kích thước của video
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Định nghĩa codec và tạo đối tượng VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Chọn codec
    video = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))

    # Ghi từng ảnh vào video
    for image in images:
        img_path = os.path.join(image_folder, image)
        video.write(cv2.imread(img_path))

    # Giải phóng tài nguyên
    video.release()
    cv2.destroyAllWindows()