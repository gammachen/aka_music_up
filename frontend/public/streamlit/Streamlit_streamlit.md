## 使用Streamlit快速搭建一个Web应用

### 构建一个config.toml配置文件

```shell
mkdir -p ~/.streamlit/

echo "\
[server] \
port = $PORT \
enableCORS = false \
headless = true \
\
" > ~/.streamlit/config.toml

export PORT=9909
bash set_streamlit_config.sh

就是说：config.toml文件内容如下：
[server]
port = $PORT
enableCORS = false
headless = true
```

```shell
streamlit run test.py
```

```shell
test.py

import streamlit as st

def blur_smile(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	smile=smile_classifier.detectMultiScale(gray, 1.6,8)
	if smile == ():
		# return img
		print('No Smile Detected')
	else:
		for x,y,w,h in smile:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img

def blur_smile_video(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	smile=smile_classifier.detectMultiScale(gray, 1.5,10)
	if smile == ():
		# return img
		print('No Smile Detected')
	else:
		for x,y,w,h in smile:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img


def about():
	st.markdown('''This Web App is made for censoring(blurring) the NSWF(Not Safe For Work) materials, that includes personal pictures depicting Nudity.
		This App will blur the inappropriate areas and body. Along with that we have provided option for blurrring Eyes, Smile, Face and Nudity.''')

	st.markdown('''YOLO custom object dedection is used for detection of Nudity whereas HAAR Cascade Classifier is used for detecting Face, Smile and Eyes.''')

def main():
	st.title("Object Detection and Masking")
	st.subheader('For Recorded as well as Real-time media')
	st.write('Using YOLOV3 object detection and Haar Cascade Classifier we detect the NSWF parts and blur them with OpenCV')

	activities = ['Home', 'About']
	choice = st.sidebar.selectbox('Select an option', activities)

	if choice == 'Home':
		st.write('Go to the about section to know more about it')

		file_type = ['Image', 'Video']
		file_choice = st.sidebar.radio('Select file type', file_type)

		if file_choice == 'Video':
			file = st.file_uploader('Choose file', ['mp4'])

			if file is not None:

				tfile = tempfile.NamedTemporaryFile(delete=False)
				tfile.write(file.read())


				st.sidebar.write('Select the required options')
				option_O = st.sidebar.checkbox('Original')
				option_E = st.sidebar.checkbox('Eyes')
				option_F = st.sidebar.checkbox('Face')
				option_S = st.sidebar.checkbox('Smile')
				option_N = st.sidebar.checkbox('Nudity')

				if st.button('Process'):
					if option_O and any([option_E, option_F, option_N, option_S]):
						st.warning('Cannot show Original and Masked image simultaneously')

					else
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							image =nudity_blur(frame)
							stframe.image(face_blur(frame))

		elif file_choice == 'Image':

			file=st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png', 'webp'])

			if file is not None:
				if file.type != 'application/pdf':
					image = Image.open(file)
					image = np.array(image)

					st.sidebar.write('Select the required options')
					option_O = st.sidebar.checkbox('Original')
					option_E = st.sidebar.checkbox('Eyes')
					option_F = st.sidebar.checkbox('Face')
					option_S = st.sidebar.checkbox('Smile')
					option_N = st.sidebar.checkbox('Nudity')


					if st.button('Process'):

						if option_O and any([option_E, option_F, option_S, option_N]):
							st.warning('Cannot show Original and Masked image simultaneously')
						else
							result_image= nudity_blur(image)
							result_image= face_blur(result_image)
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)					
	
	elif choice =='About':
		about()

if __name__ == '__main__':
	main()
```