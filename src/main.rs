use opencv::{core, highgui, imgproc, objdetect, prelude::*, types, videoio, Result};
fn main() -> Result<()> {
    //println!("Hello, world!");
    let mut camera = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let mut xml_path = "/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml";
    let mut face_detector = objdetect::CascadeClassifier::new(xml_path)?;
    let mut img = Mat::default();
    loop {
        camera.read(&mut img)?;
        let mut gray = Mat::default();
        imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        let mut faces = types::VectorOfRect::new();
        face_detector.detect_multi_scale(
            &gray,
            &mut faces,
            1.1,
            10,
            objdetect::CASCADE_SCALE_IMAGE,
            core::Size::new(10, 10),
            core::Size::new(0, 0),
        )?;
        
        if faces.len()>0 {
            for face in faces.iter() {
                println!("{:?}", face);
                imgproc::rectangle(
                    &mut img,
                    face,
                    core::Scalar::new(0f64, 255f64, 0f64, 0f64),
                    2,
                    imgproc::LINE_8,
                    0,
                )?;
            }
        }

        highgui::imshow("gray", &img)?;
        highgui::wait_key(1)?;
    }
    Ok(())
}
