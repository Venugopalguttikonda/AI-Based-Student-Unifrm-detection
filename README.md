Project Title: AI-Based Student Uniform Detection System

Technologies Used: YOLOv11, OpenCV, Dlib, Python, SMTP (for email notifications)

---

 Project Overview

This project is an AI-powered system developed for **automated student uniform detection** using state-of-the-art computer vision techniques. The primary objective of this system is to **monitor students in educational institutions** and determine whether they are wearing the prescribed uniform.

If a student is detected not wearing the uniform, the system **automatically sends an email alert with a fine** notification to the registered email ID of the student.

---

 Core Technologies

* **YOLOv11** (You Only Look Once, Version 11): Used for real-time object detection to identify whether the student is wearing a uniform or not.
* **OpenCV**: Used for handling image/video processing and webcam integration.
* **Dlib**: Used for facial recognition and feature extraction to identify students.
* **SMTP / smtplib**: Used to send email alerts automatically to students when a uniform violation is detected.

---
 How the System Works

1. Student Registration & Face Capture:

   * Students must first register themselves by providing their **Name**, **Email ID**, and **capturing their face image**.
   * This information is saved in the system to enable **face recognition and identification** during uniform checks.

2. Uniform Detection Process:

   * Once student registration is complete, the system uses YOLOv11 to analyze the live webcam feed or video footage.
   * It detects whether the student is in **uniform or non-uniform** based on trained AI models.

3. Violation Handling:

   * If a student is detected **not wearing the uniform**, the system identifies the student using facial recognition.
   * An automated email is sent to the student’s registered email ID, notifying them of the violation and imposing a fine.

---

 How to Run the Project

 Step 1: Register Student and Capture Face

Run the following Python script:

```bash
python face_capture.py
```

This will:

* Prompt the user to enter the student’s **name and email**.
* Activate the **webcam** to capture multiple images of the student’s face for training and recognition.
* Store the data in the respective database/directory for future detection.

---

Step 2: Run Uniform Detection

Run the main detection system:

```bash
python main.py
```

This will:

* Activate the **webcam** or video feed.
* Perform **face recognition** using Dlib to identify the student.
* Use **YOLOv11 and OpenCV** to analyze whether the student is wearing a uniform.
* If **non-uniform** is detected:

  * The system sends an **email alert** to the student with a fine notice.

---

Sample Email Alert Sent

> Subject: Uniform Violation Notice
> Body:
> Dear \[Student Name],
>
> Our system detected that you were not wearing the proper school uniform today. As per institutional guidelines, this is a violation. A fine has been imposed.
>
> Kindly ensure compliance with the dress code from the next day onward.
>
> Regards,
> Admin Team

---

Future Enhancements

* Integration with attendance system*
* Generation of **violation reports for administrators**.
* **Mobile app interface** for parents and students.
* Real-time monitoring using **CCTV feeds**.
* Use of **Deep Learning models** for improved uniform detection accuracy.

---

