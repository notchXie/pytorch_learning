import dicom2nifti

if __name__ == '__main__':
    dicom_path = r"C:\Users\notch\Desktop\DISCOM"
    dicom2nifti.convert_directory(dicom_path,'./')


