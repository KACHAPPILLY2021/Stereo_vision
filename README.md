<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">


  <h1 align="center">Stereo Vision </h1>


</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary><h3>Table of Contents</h3></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#demo">Demo</a></li>
      </ul>
    </li>
    <li>
      <a href="#pipeline">Pipeline</a>
      <ul>
        <li><a href="#calibration">Calibration</a></li>
	<li><a href="#report">Rectification</a></li>
	<li><a href="#correspondance">Correspondance</a></li>
	<li><a href="#compute-depth-image">Compute Depth Image</a></li>
      </ul>
    </li>
    <li>
      <a href="#report">Report</a>
      <ul>
        <li><a href="#dataset">Dataset</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#usage">Usage</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project



Implemented Stereo Vision on three datasets, each with two images of the same scenario from different camera angles. Analyzed relative object positions in both images to obtain 3D information. The process involves comparing the visual data from two vantage points, enabling the extraction of rich 3D information that can be used for various applications.

Found disparity and depth map of two image sequences of a given subject by leveraging the concepts of **Epipolar Geometry**, **Fundamental Matrix** (F), **Essential Matrix** (E) and its decomposition to get Rotation and Translation matrices, epipolar lines, rectification correspondence using SSD.

 Note that ```no OpenCV inbuilt``` function was used while implementing these concepts .


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Demo

<div align="center">


  <h4 align="center"> Features matching in left and right images</h4>


</div>

<img src="https://github.com/KACHAPPILLY2021/Stereo_vision/blob/main/output/Matches_screenshot_18.04.2022.png?raw=true" height="255" width="1000" alt="features">
<div align="center">


  <h4 align="center">Epipolar line corrosponding to the obtained matching features</h4>


</div>

<img src="https://github.com/KACHAPPILLY2021/Stereo_vision/blob/main/output/epipolar%20(1).png?raw=true" height="280" width="1000" alt="epipolar">
<div align="center">


  <h4 align="center">Rectified images</h4>


</div>

<img src="https://github.com/KACHAPPILLY2021/Stereo_vision/blob/main/output/rectified.jpg?raw=true" height="255" width="1000" alt="rectified">
<div align="center">


  <h4 align="center"> Depth and Disparity map</h4>


</div>

<img src="https://github.com/KACHAPPILLY2021/Stereo_vision/blob/main/output/depth_disparity.jpg?raw=true" height="255" width="1000" alt="map">
<div align="center">


  <h4 align="center"> Depth and Disparity heat maps</h4>


</div>

<img src="https://github.com/KACHAPPILLY2021/Stereo_vision/blob/main/output/heatmap.jpg?raw=true" height="255" width="1000" alt="heat map">
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Pipeline -->
## Pipeline

### Calibration

1) Compared the two images and selected the set of matching features. Tuned the Lowe's ration to reject the outliers
2) Estimated the Fundamental matrix using the obtained matching feature and enforced rank 2. 
3) Estimated the Essential matrix(E) from the Fundamental matrix(F) and instrinsic camera parameter.
4) Decomposed the E into a translation T and rotation R.

### Rectification

1) Applied perspective transfomation to make sure that the epipolar lines are horizontal for both the images. This will limit the search space to horizontal line during the corrospondace matching process in the later stage.

### Correspondance

1) For each epipolar line, sliding window technique with SSD was applied to find the corrospondence and calulate disparity.
2) Rescaled the disparity to be from 0-255 for visualization.

### Compute Depth Image

1) Using the disparity calculated above, the depth map was computed. The resultant image has a depth image instead of disparity.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Reports -->
## Report

The detailed report for this project can be found here. [Report](https://github.com/KACHAPPILLY2021/Stereo_vision/blob/main/ENPM_673_P3.pdf)

### Dataset

[MiddleBury Stereo Dataset](https://vision.middlebury.edu/stereo/data/scenes2021/#description)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

These are the instructions to get started on the project.
To get a local copy up and running follow these simple steps.

### Prerequisites
* Python 3.8.10
* Libraries - OpenCV, Numpy, matplotlib
* OS - Linux (tested)


### Usage

1. Clone the repo
   ```sh
   git clone https://github.com/KACHAPPILLY2021/Stereo_vision.git
   ```
2. Open the folder ```Stereo_vision``` in IDE and RUN each file or navigate using terminal
   ```sh
   cd ∼ /Stereo_vision
   ```
3. To run program which display outputs like shown above :
   ```sh
   python3 curule.py
   ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Jeffin Johny K - [![MAIL](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:jeffinjk@umd.edu)
	
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/KACHAPPILLY2021)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](http://www.linkedin.com/in/jeffin-johny-kachappilly-0a8597136)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See [MIT](https://choosealicense.com/licenses/mit/) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
