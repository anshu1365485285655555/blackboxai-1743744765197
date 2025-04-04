
Built by https://www.blackbox.ai

---

```markdown
# AI Design Generator

## Project Overview
The **AI Design Generator** is a web application that allows users to upload images and generate enhanced design variations using artificial intelligence. The tool supports JPG and PNG formats and provides a user-friendly interface for uploading images, processing them, and displaying the results with various design styles.

## Installation
To run the AI Design Generator locally, follow these steps:
1. Clone this repository:
    ```bash
    git clone <repository-url>
    ```
2. Navigate to the project directory:
    ```bash
    cd ai-design-generator
    ```
3. Open `index.html` in your preferred browser.

*Note: Ensure you have a local server running if you want to handle file uploads and processing.*

## Usage
1. Open the `index.html` file in a web browser.
2. Drag and drop an image into the designated area or click the "Browse Files" button to upload an image (JPG or PNG, maximum size 5MB).
3. Once the image is uploaded, you'll see a preview of it along with options to change it.
4. Click the "Generate Designs" button to initiate the AI processing.
5. After processing, you will be redirected to the results page showcasing various design variations which can be downloaded.

## Features
- **Responsive Design**: The app adapts to different screen sizes.
- **Image Upload**: Supports drag and drop functionality as well as file browsing.
- **AI Processing**: Generates design variations based on the uploaded image.
- **Real-time Feedback**: Displays progress during design generation with an interactive progress bar.
- **Download Options**: Users can download the generated designs directly from the results page.
- **Toast Notifications**: Provides real-time user feedback on actions and errors.

## Dependencies
This project utilizes the following libraries:
- **Tailwind CSS**: For styling the application components.
- **Font Awesome**: For icons used throughout the application.

No specific packages are defined in `package.json` as this project is primarily HTML/CSS/JavaScript and does not require any external Node.js dependencies.

## Project Structure
```plaintext
.
├── index.html         # Main page for uploading images
├── process.html       # Page displayed during image processing
├── results.html       # Page displaying generated design variations
├── script.js          # JavaScript handling file upload, processing logic, and UI interactions
├── styles.css         # Custom styles for the application
```

## License
This project is licensed under the MIT License.
```