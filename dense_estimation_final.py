import numpy as np
import cv2
import copy
from make_video import make_video
from tqdm import tqdm
videoname='diningroom'#视频名称
method='MOG'#使用的方法
filename=videoname+'-'+method

def main():
    capture = cv2.VideoCapture(videoname+'.mp4')
    #background_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))#获取视频总帧数

    warning_threshold=15#报警阈值
    print("Start processing...")
    first_iteration_indicator = 1
    threshold = 2
    maxValue = 2
    for i in tqdm(range(0, length), ncols=100):#进度条总长度为100

        ret, frame = capture.read()
        frame = cv2.flip(frame, -1)

        # If first frame
        if first_iteration_indicator == 1:

            first_frame = copy.deepcopy(frame)
            global height, width#图片的高度和宽度
            height, width = frame.shape[:2]
            global height_size,width_size
            height_size = height // 10#分割成的小区域的宽度和高度
            width_size = width // 10
            accum_image = np.zeros((height, width), np.uint8)
            global size2
            size2 = 2 * height_size * width_size
            first_iteration_indicator = 0

        else:
            filter = background_subtractor.apply(frame)  # remove the background
            cv2.imwrite(filename+'-frame_test_dining.jpg', frame)
            cv2.imwrite(filename+'-diff-bkgnd-frame_test_dining.jpg', filter)
            ret, th1 = cv2.threshold(filter, threshold, maxValue, cv2.THRESH_BINARY)#黑白二值化

            # add to the accumulated image
            accum_image = cv2.add(accum_image, th1)
            cv2.imwrite(filename+'-mask_test_dining.jpg', accum_image)
            color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_SUMMER)#伪彩色
            video_frame = cv2.addWeighted(frame, 0.7, color_image_video, 0.5, 0)#按权重叠加

            #cv2.imwrite(name, video_frame)
            if i % 10 == 1:
                number,max_i,max_j=dense_max(accum_image)
                text = "Cumulative traffic:" + str(number)  # 待添加的文字
                text_color=(0, 255, 0)

                if number>warning_threshold:
                    text = text + '  danger!'
                    text_color = (0,0,255)
                    warning_area = np.zeros((height, width,3), np.uint8)
                    for x in range(max_i * width_size, (max_i + 1) * width_size):
                        for y in range(max_j * height_size, (max_j + 1) * height_size):
                            pixel_value= accum_image[y, x]
                            warning_area[y, x] = (pixel_value,0,0)

                    video_frame=cv2.add(video_frame,warning_area)

            text_position = (int((max_i + 0.5) * width_size), int((max_j + 0.5) * height_size))
            cv2.putText(video_frame,text,text_position,cv2.FONT_HERSHEY_TRIPLEX, 0.8, text_color, 2, cv2.LINE_AA)
            name = "./frames_test_dining/frame%d.jpg" % i
            cv2.imwrite(name, video_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    make_video('./frames_test_dining/', filename+'-output.avi')
    np.savetxt(filename+'.txt',accum_image,fmt='%d')#将数组保存到文件中，方便后续调试
    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

     # save the final heatmap
    cv2.imwrite(filename+'-diff-overlay_test.jpg', result_overlay)
    dense_every(accum_image)
    # cleanup
    capture.release()
    cv2.destroyAllWindows()
def dense_max(dense_image):
#该函数的作用是从累加的人流密度图中找出各区域最大值并返回最大值区域位置和密度估计值
#函数的输入是一个数组
    max=0
    max_i = 0
    max_j = 0
    for i in range(10):
        for j in range(10):
            region=dense_image[i*width_size:(i+1)*width_size,j*height_size:(j+1)*height_size]
            sum=region.sum()
            if(sum>max):
                max=sum
                max_i=i
                max_j=j

    number=round(max/size2)

    return number,max_i,max_j

def dense_every(image):
#估算10*10个区域的人流密度，并在图中显示人流密度的估计值，保存图片
    bk_img = cv2.imread(filename+"-diff-overlay_test.jpg")

    for i in range(10):
        for j in range(10):
            region=image[i*width_size:(i+1)*width_size,j*height_size:(j+1)*height_size]
            sum=region.sum()
            cv2.putText(bk_img, str(int(sum/size2)), (int((i+0.5)*width_size), int((j+0.5)*height_size)), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 1, cv2.LINE_AA)#在图片中放置人流密度的估计值

    cv2.imwrite(filename+"-add_text.jpg",bk_img)

if __name__ == '__main__':
    main()