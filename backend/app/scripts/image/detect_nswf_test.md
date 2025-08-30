```shell
(base) shhaofu@shhaofudeMacBook-Pro NSWF-Nudity-Blurring-System % streamlit run test.py
2025-03-29 08:55:04.940
Warning: the config option 'server.enableCORS=false' is not compatible with 'server.enableXsrfProtection=true'.
As a result, 'server.enableCORS' is being overridden to 'true'.

More information:
In order to protect against CSRF attacks, we send a cookie with each request.
To do so, we must specify allowable origins, which places a restriction on
cross-origin resource sharing.

If cross origin resource sharing is required, please disable server.enableXsrfProtection.


  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:9909
  Network URL: http://192.168.31.108:9909
  External URL: http://220.246.100.107:9909

  For better performance, install the Watchdog module:

  $ xcode-select --install
  $ pip install watchdog

────────────────────────── Traceback (most recent call last) ───────────────────────────
  /opt/anaconda3/lib/python3.11/site-packages/streamlit/runtime/scriptrunner/exec_code
  .py:121 in exec_func_with_error_handling

  /opt/anaconda3/lib/python3.11/site-packages/streamlit/runtime/scriptrunner/script_ru
  nner.py:640 in code_to_exec

  /Users/shhaofu/Code/Codes/NSWF-Nudity-Blurring-System/test.py:510 in <module>

    507 │   │   about()
    508
    509 if __name__ == '__main__':
  ❱ 510 │   main()
    511
    512
    513

  /Users/shhaofu/Code/Codes/NSWF-Nudity-Blurring-System/test.py:465 in main

    462 │   │   │   │   │   │
    463 │   │   │   │   │   │
    464 │   │   │   │   │   │   elif option_E and option_S:
  ❱ 465 │   │   │   │   │   │   │   result_image = blur_eyes(blur_smile(image))
    466 │   │   │   │   │   │   │   st.image(result_image, use_column_width=True)
    467 │   │   │   │   │   │   │   pil_img = Image.fromarray(result_image)
    468 │   │   │   │   │   │   │   st.markdown(get_image_download_link(pil_img), unsa

  /Users/shhaofu/Code/Codes/NSWF-Nudity-Blurring-System/test.py:80 in blur_smile

     77 def blur_smile(img):
     78 │   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     79 │   smile=smile_classifier.detectMultiScale(gray, 1.6,8)
  ❱  80 │   if smile == ():
     81 │   │   # return img
     82 │   │   print('No Smile Detected')
     83 │   else:
────────────────────────────────────────────────────────────────────────────────────────
ValueError: operands could not be broadcast together with shapes (2,4) (0,)
────────────────────────── Traceback (most recent call last) ───────────────────────────
  /opt/anaconda3/lib/python3.11/site-packages/streamlit/runtime/scriptrunner/exec_code
  .py:121 in exec_func_with_error_handling

  /opt/anaconda3/lib/python3.11/site-packages/streamlit/runtime/scriptrunner/script_ru
  nner.py:640 in code_to_exec

  /Users/shhaofu/Code/Codes/NSWF-Nudity-Blurring-System/test.py:510 in <module>

    507 │   │   about()
    508
    509 if __name__ == '__main__':
  ❱ 510 │   main()
    511
    512
    513

  /Users/shhaofu/Code/Codes/NSWF-Nudity-Blurring-System/test.py:492 in main

    489 │   │   │   │   │   │   │   st.markdown(get_image_download_link(pil_img), unsa
    490 │   │   │   │   │   │
    491 │   │   │   │   │   │   elif option_S:
  ❱ 492 │   │   │   │   │   │   │   result_image= blur_smile(image)
    493 │   │   │   │   │   │   │   st.image(result_image, use_column_width=True)
    494 │   │   │   │   │   │   │   pil_img = Image.fromarray(result_image)
    495 │   │   │   │   │   │   │   st.markdown(get_image_download_link(pil_img), unsa

  /Users/shhaofu/Code/Codes/NSWF-Nudity-Blurring-System/test.py:80 in blur_smile

     77 def blur_smile(img):
     78 │   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     79 │   smile=smile_classifier.detectMultiScale(gray, 1.6,8)
  ❱  80 │   if smile == ():
     81 │   │   # return img
     82 │   │   print('No Smile Detected')
     83 │   else:
────────────────────────────────────────────────────────────────────────────────────────
ValueError: operands could not be broadcast together with shapes (2,4) (0,)
```