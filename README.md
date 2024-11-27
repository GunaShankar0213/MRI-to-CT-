# MRI-to-CT Conversion Model

This repository contains a **CycleGAN-based MRI-to-CT conversion model**, designed to reduce the need for radiation-intensive CT scans by generating CT-equivalent images from MRI scans. The project leverages state-of-the-art models such as **CycleGAN**, **Pix2Pix**, and **U-Net with Attention Layers** to achieve high-quality MRI-to-CT image translation.

---

## Project Overview

Medical imaging plays a crucial role in diagnostics and treatment planning. While CT scans provide detailed bone and tissue imaging, they expose patients to significant radiation. MRI scans, on the other hand, are safer but lack the same level of detail for specific clinical requirements. This project aims to bridge that gap by using deep learning to synthesize CT-equivalent images from MRI scans.

The **MRI-to-CT Conversion Model** offers:
- **Non-invasive imaging alternatives**: Reducing patient exposure to harmful radiation.
- **Enhanced diagnostic support**: Generating CT-equivalent images with high fidelity.
- **Versatile adaptability**: Supports multiple architectures tailored for specific medical imaging needs.

---

## Models Used

### 1. **CycleGAN**
- CycleGAN is utilized to enable unpaired image-to-image translation between MRI and CT domains.
- Features:
  - Cyclic consistency loss ensures the MRI-to-CT and CT-to-MRI translations are accurate.
  - Adaptable for datasets without strict one-to-one pairings.

### 2. **Pix2Pix**
- Pix2Pix model is employed for paired MRI-to-CT image translation.
- Features:
  - Conditional GAN architecture ensures high-quality generation.
  - Fine-grained control over feature mapping.

### 3. **U-Net with Attention Layers**
- U-Net architecture, enhanced with attention mechanisms, improves the focus on relevant regions in medical images.
- Features:
  - Handles complex anatomical structures with precision.
  - Efficient segmentation and translation capabilities.

---

## Workflow

1. **Data Preprocessing**:
   - Paired and unpaired datasets of MRI and CT images are normalized and augmented for training.
   - MRI images serve as input while CT images are the target domain.

![image](https://github.com/user-attachments/assets/9091af2a-b63f-413c-9a45-c0c42058d71d)

2. **Model Training**:
   - CycleGAN is trained for unpaired datasets.
   - Pix2Pix is fine-tuned for paired datasets.
   - U-Net with attention layers is used to enhance region-specific details.

3. **Inference**:
   - The trained models convert MRI scans into CT-like images, maintaining diagnostic quality and resolution.

---

## Future Applications

- **Clinical Diagnostics**:
  - Offers a safer alternative to CT scans, particularly for repeat imaging in conditions like cancer monitoring.
  
- **Radiation-Free Imaging**:
  - Enables CT-level imaging details without exposing patients to radiation.

- **Multi-Modal Imaging Studies**:
  - Facilitates research in combining MRI and CT imaging data for enhanced diagnostic insights.

- **Telemedicine**:
  - Streamlines remote diagnostics by generating multiple imaging modalities from a single MRI.

---

## Efficiency and Performance

- High-quality image generation with minimal loss of diagnostic details.
- U-Net with attention layers improves the accuracy and focus on critical regions, making it ideal for complex anatomical structures.
- CycleGAN ensures robust performance even with limited paired datasets.
- Supports real-time inference on GPU-accelerated environments.

---

## Weights and Reusability

The pre-trained weights provided in this repository can be used to:
- Directly convert MRI images to CT-like images.
- Fine-tune models for domain-specific datasets.
- Integrate into clinical workflows for rapid imaging solutions.

---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/MRI-to-CT.git
   cd MRI-to-CT
   ```

2. **Setup the environment**:
   Install dependencies using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Weights**:
   - Link to weights will be provided in the `weights/` directory (drive).

---
##Predicted sample 

![image](https://github.com/user-attachments/assets/acfdb033-ffbe-4b1f-bedc-05ea57bd9d3d)

![predicted_ct_image_1](https://github.com/user-attachments/assets/af4f3722-3dc2-445e-b77a-6dd25babdc43)

![predicted_ct_image_2](https://github.com/user-attachments/assets/fd1165c9-a4c3-4ee4-b54b-5f138314805c)


## Future Enhancements

- **Multi-scale attention mechanisms** for improved context understanding.
- **Integration with federated learning** for collaborative training across hospitals.
- **Real-time deployment** on medical-grade devices for clinical use.

---

## Acknowledgments

This project leverages research and innovations in medical imaging and deep learning. A special thanks to the developers of CycleGAN, Pix2Pix, and U-Net for their open-source contributions.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
