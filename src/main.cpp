#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <map>
#include <set>
#include <regex>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

// --- Helper Functions (unchanged) ---
Pix* mat8ToPix(const cv::Mat& mat);
cv::Mat pixToMat(Pix* pix);
std::string post_process_ocr_text(const std::string& text);

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
        return 1;
    }

    fs::path directory_path = argv[1];
    if (!fs::is_directory(directory_path)) {
        std::cerr << "Error: '" << directory_path << "' is not a valid directory." << std::endl;
        return 1;
    }
    
    fs::path debug_output_dir = "debug_output";
    if (!fs::exists(debug_output_dir)) {
        fs::create_directory(debug_output_dir);
    }

    std::ofstream output_file("results.csv");
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open 'results.csv' for writing." << std::endl;
        return 1;
    }

    tesseract::TessBaseAPI* ocr = new tesseract::TessBaseAPI();
    // Initialize Tesseract for both English and Mongolian
    if (ocr->Init(nullptr, "eng+mon", tesseract::OEM_LSTM_ONLY)) {
        std::cerr << "Could not initialize Tesseract OCR engine." << std::endl;
        output_file.close();
        return 1;
    }
    
    // Set PageSegMode for reading a single block of text
    ocr->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK); 


    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.is_regular_file()) {
            std::string file_extension = entry.path().extension().string();
            std::transform(file_extension.begin(), file_extension.end(), file_extension.begin(), ::tolower);

            if (file_extension == ".jpg" || file_extension == ".jpeg" || file_extension == ".png") {
                std::string image_path = entry.path().string();
                std::string base_filename = entry.path().stem().string();
                std::cout << "\nProcessing: " << image_path << std::endl;

                cv::Mat original_image = cv::imread(image_path, cv::IMREAD_COLOR);
                if (original_image.empty()) {
                    std::cerr << "  Error: Could not open or find image: " << image_path << std::endl;
                    continue;
                }
                if (original_image.cols <= 0 || original_image.rows <= 0) {
                     std::cerr << "  Error: Image has zero or negative dimensions after loading: " << image_path << std::endl;
                     continue;
                }

                Pix* pix_original = nullptr;
                Pix* pix_binarized = nullptr;
                Pix* pix_temp = nullptr;

                pix_original = mat8ToPix(original_image);
                if (!pix_original) {
                    std::cerr << "  Error: Failed to convert OpenCV Mat to Leptonica Pix for: " << image_path << std::endl;
                    continue;
                }
                
                // --- Preprocessing remains the same, omitted for brevity ---
                // a. DPI/Scaling
                l_int32 current_xres = pixGetXRes(pix_original);
                l_int32 current_yres = pixGetYRes(pix_original);
                l_float32 scale_factor = 1.0;
                const l_int32 TARGET_DPI = 300;
                if (current_xres == 0 || current_yres == 0) {
                    current_xres = (current_xres == 0) ? 72 : current_xres;
                    current_yres = (current_yres == 0) ? 72 : current_yres;
                }
                if (current_xres < TARGET_DPI || current_yres < TARGET_DPI) {
                    scale_factor = std::max((l_float32)TARGET_DPI / current_xres, (l_float32)TARGET_DPI / current_yres);
                    if (scale_factor > 1.0) {
                        pix_temp = pixScale(pix_original, scale_factor, scale_factor);
                        pixDestroy(&pix_original);
                        pix_original = pix_temp;
                        pix_temp = nullptr;
                    }
                }
                pixSetXRes(pix_original, TARGET_DPI);
                pixSetYRes(pix_original, TARGET_DPI);

                // b. Grayscale Conversion
                if (pixGetDepth(pix_original) != 8) {
                    pix_temp = pixConvertRGBToLuminance(pix_original);
                    pixDestroy(&pix_original);
                    pix_original = pix_temp;
                }

                // c. Deskewing
                l_float32 angle, conf;
                pix_temp = pixFindSkewAndDeskew(pix_original, 0, &angle, &conf);
                if (pix_temp) {
                    pixDestroy(&pix_original);
                    pix_original = pix_temp;
                }

                // d. Adaptive Binarization
                l_ok otsu_status = pixOtsuAdaptiveThreshold(pix_original, 200, 200, 10, 10, 0.1f, nullptr, &pix_binarized);
                if (otsu_status != 0 || !pix_binarized) { 
                    pixDestroy(&pix_original); 
                    continue;
                }
                pixDestroy(&pix_original); 
                
                // e. Noise Removal
                pix_temp = pixOpenBrick(nullptr, pix_binarized, 2, 2); 
                if (pix_temp) {
                    pixDestroy(&pix_binarized);
                    pix_binarized = pix_temp;
                }

                // --- Single-Region OCR ---
                // Define the region to OCR
                int x_start = 2250;
                int x_end = 3400;

                int y_start = 250;
                int y_end = pixGetHeight(pix_binarized);
                
                // Add a small margin to the cropped region for stability
                int margin = 5;
                int crop_x = x_start + margin;
                int crop_y = y_start + margin;
                int crop_width = (x_end - x_start) - (2 * margin);
                int crop_height = (y_end - y_start) - (2 * margin);

                if (crop_width <= 0 || crop_height <= 0) {
                    std::cerr << "  Error: Invalid crop dimensions for region. Skipping." << std::endl;
                    pixDestroy(&pix_binarized);
                    continue;
                }
                
                Box* box = boxCreate(crop_x, crop_y, crop_width, crop_height);
                Pix* pix_region = pixClipRectangle(pix_binarized, box, nullptr);
                boxDestroy(&box);
                pixDestroy(&pix_binarized);

                if (!pix_region) {
                    std::cerr << "  Error: Failed to crop the OCR region. Skipping." << std::endl;
                    continue;
                }
                
                cv::Mat debug_region_mat = pixToMat(pix_region);
                cv::imwrite( (debug_output_dir / (base_filename + "_ocr_region.png")).string(), debug_region_mat);

                ocr->SetImage(pix_region);
                char* outText = ocr->GetUTF8Text();
                std::string raw_text = (outText != nullptr) ? outText : "";
                delete[] outText;
                pixDestroy(&pix_region);

                std::cout << "  Raw OCR text length: " << raw_text.length() << std::endl;
                // --- Post-processing ---
                std::string processed_text = post_process_ocr_text(raw_text);

                // --- Output Results ---
                std::cout << "\n--- Processed OCR Result for " << entry.path().filename().string() << " ---\n";
                std::cout << processed_text << std::endl;
                std::cout << "--------------------------------------------------\n";

                // Save to file, but only if there's content to save
                if (!processed_text.empty()) {
                   // output_file << "--- " << entry.path().filename().string() << " ---\n";
                    output_file << processed_text << "\n\n";
                }
                // Save to file
                //output_file << "--- " << entry.path().filename().string() << " ---\n";
               // output_file << raw_text << "\n\n";
            }
        }
    }

    ocr->End();
    delete ocr;
    output_file.close();
    std::cout << "\nOCR processing complete. Results saved to 'results.csv'." << std::endl;

    return 0;
}


// Function to convert OpenCV Mat to Leptonica Pix format
Pix* mat8ToPix(const cv::Mat& mat) {
    if (mat.empty()) {
        std::cerr << "Debug: mat8ToPix received empty Mat." << std::endl;
        return nullptr;
    }
    if (mat.cols <= 0 || mat.rows <= 0) {
        std::cerr << "Debug: mat8ToPix received Mat with non-positive dimensions: "
                  << mat.cols << "x" << mat.rows << std::endl;
        return nullptr;
    }

    cv::Mat temp_mat = mat.clone(); // Work on a copy
    if (temp_mat.depth() != CV_8U) {
        std::cerr << "Error: Input Mat must be 8-bit unsigned char. Attempting conversion." << std::endl;
        temp_mat.convertTo(temp_mat, CV_8U);
    }

    // Convert to grayscale if it's a color image
    if (temp_mat.channels() == 3 || temp_mat.channels() == 4) {
        cv::cvtColor(temp_mat, temp_mat, cv::COLOR_BGR2GRAY);
    }

    std::vector<uchar> buf;
    bool encode_success = cv::imencode(".png", temp_mat, buf);
    if (!encode_success) {
        std::cerr << "Debug: cv::imencode failed for the Mat." << std::endl;
        return nullptr;
    }
    if (buf.empty()) {
        std::cerr << "Debug: Encoded buffer is empty, size = 0." << std::endl;
        return nullptr;
    }

    Pix* pix_result = pixReadMem(buf.data(), buf.size());
    if (!pix_result) {
        std::cerr << "Debug: pixReadMem failed to create Pix from buffer." << std::endl;
    }
    return pix_result;
}

// Function to convert Leptonica Pix to OpenCV Mat
cv::Mat pixToMat(Pix* pix) {
    if (!pix) {
        return cv::Mat();
    }

    l_int32 width = pixGetWidth(pix);
    l_int32 height = pixGetHeight(pix);
    l_int32 depth = pixGetDepth(pix);

    if (depth == 1) { // Binary image (1 bpp)
        cv::Mat mat(height, width, CV_8UC1);
        for (l_int32 y = 0; y < height; ++y) {
            l_uint32* line = pixGetData(pix) + y * pixGetWpl(pix);
            for (l_int32 x = 0; x < width; ++x) {
                mat.at<uchar>(y, x) = GET_DATA_BIT(line, x) ? 255 : 0; // 1 (white) -> 255, 0 (black) -> 0.
            }
        }
        return mat;
    } else if (depth == 8) { // Grayscale image (8 bpp)
        cv::Mat mat(height, width, CV_8UC1);
        for (l_int32 y = 0; y < height; ++y) {
            l_uint32* line = pixGetData(pix) + y * pixGetWpl(pix);
            for (l_int32 x = 0; x < width; ++x) {
                mat.at<uchar>(y, x) = GET_DATA_BYTE(line, x);
            }
        }
        return mat;
    } else if (depth == 32) { // Color image (32 bpp - ARGB or RGBA)
        cv::Mat mat(height, width, CV_8UC4); // OpenCV typically uses BGRA for 4-channel
        for (l_int32 y = 0; y < height; ++y) {
            l_uint32* line = pixGetData(pix) + y * pixGetWpl(pix);
            for (l_int32 x = 0; x < width; ++x) {
                l_uint32 pixel_val = *(line + x);
                l_int32 r, g, b, a;
                
                extractRGBAValues(pixel_val, &r, &g, &b, &a); 
                
                mat.at<cv::Vec4b>(y, x)[0] = static_cast<uchar>(b); // Blue
                mat.at<cv::Vec4b>(y, x)[1] = static_cast<uchar>(g); // Green
                mat.at<cv::Vec4b>(y, x)[2] = static_cast<uchar>(r); // Red
                mat.at<cv::Vec4b>(y, x)[3] = static_cast<uchar>(a); // Alpha
            }
        }
        cv::Mat bgr_mat;
        cv::cvtColor(mat, bgr_mat, cv::COLOR_RGBA2BGR);
        return bgr_mat;
    }
    std::cerr << "Warning: pixToMat does not support Pix depth " << depth << std::endl;
    return cv::Mat();
}

// Function to post-process the raw OCR string by removing empty lines
std::string post_process_ocr_text(const std::string& text) {
    std::string cleaned_text;
    std::istringstream iss(text);
    std::string line;
    
    // Iterate through each line of the input text
    while (std::getline(iss, line)) {
        // Trim leading and trailing whitespace from the line
        // Note: find_first_not_of returns the position of the first non-whitespace character
        // If no such character is found, it returns std::string::npos, so we handle that case.
        const auto first_char = line.find_first_not_of(" \t\r\n");
        if (std::string::npos == first_char) {
            continue; // The line is empty or just whitespace
        }
        const auto last_char = line.find_last_not_of(" \t\r\n");
        
        // Extract the trimmed line
        const std::string trimmed_line = line.substr(first_char, (last_char - first_char + 1));

        // If the line is not empty after trimming, add it to the cleaned text
        if (!trimmed_line.empty()) {
            cleaned_text += trimmed_line + "\n";
        }
    }
    
    // Remove the final newline character if it exists
    if (!cleaned_text.empty() && cleaned_text.back() == '\n') {
        cleaned_text.pop_back();
    }
    
    return cleaned_text;
}


