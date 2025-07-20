# TrailTracker

A web application for managing and analyzing trail camera footage.

## TODO
- [x] Add pop-ups for incorrect password/username
- [x] Make cameras show longitude and latitude
- [x] Display actual photos instead of icons (in profile)
- [ ] Parse info differently based on camera ID presence
- [ ] Review all features and verify functionality
- [ ] Sort by date
- [ ] Make run over HTTPS instead of HTTP (need domain name)
- [ ] Get pin of current location working in online version(should work when getting domain name)


### Mobile Development
- [ ] Create React Native version
- [ ] Connect to same Node backend

### AI Improvements
- [ ] Train AI on additional footage
- [ ] Explore better training methods
- [ ] Improve detection of other animals
- [ ] Expand animal detection array

### Security Enhancements
- [ ] Add more security measures
- [ ] Implement password requirements
- [ ] Add email verification 

#######################################################################################################################
PROFILE STYLING:
<style>
        body, html {
            height: 100%;
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: url("{{ url_for('static', filename='background.png') }}") no-repeat center center fixed;
            background-size: cover;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            overflow: hidden; /* Prevent scrolling on body */
        }

        .container {
            background-color: rgba(50, 50, 50, 0.8);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            width: 90%; /* Ensure a small margin on the sides */
            text-align: center;
            color: white;
            margin-top: 60px; /* Add margin to avoid overlap with the top bar */
            margin-bottom: 60px; /* Add margin to avoid overlap with the bottom bar */
            overflow-y: auto; /* Enable vertical scrolling */
            position: relative;
        }

        .top-bar, .bottom-bar {
            position: fixed;
            left: 0;
            width: 100%;
            height: 50px;
            background-color: #323232;
            display: flex;
            align-items: center;
            justify-content: center; /* Center align the items */
            z-index: 1000;
        }

        .top-bar {
            top: 0;
        }

        .bottom-bar {
            bottom: 0;
        }

        .title {
            color: white;
            font-size: 24px;
            margin: 0;
            text-align: center;
            display: flex;
            justify-content: center; /* Center align the text */
            align-items: center; /* Center vertically */
        }

        .nav-btn {
            background-color: #323232;
            color: white;
            border: none;
            cursor: pointer;
            padding: 10px;
            text-align: center;
            margin: 0 40px; /* Add margin for spacing between buttons */
            border-radius: 50%; /* Make hover shape round */
        }

        .nav-btn:hover {
            background-color: #444;
        }

        .nav-btn img {
            width: 25px;
            height: 25px;
        }

        .profile-pic {
            border-radius: 50%;
            width: 150px;
            height: 150px;
            object-fit: cover;
            display: block;
            margin: 0 auto 10px;
        }

        .upload-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }

        .upload-details {
            flex-grow: 1;
        }

        .upload-item {
            display: flex;
            align-items: center;
            justify-content: flex-start; /* Ensure content starts from the left */
            margin-bottom: 20px;
        }

        .upload-details {
            flex-grow: 1;
            margin-right: 10px; /* Add space between text and icon */
        }

        .animal-icon {
            width: 50px;
            height: 50px;
        }

        .btn-container {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }


        .input, .btn {
            margin-bottom: 15px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            width: 90%; /* Reduce width for a narrower form */
            margin: 0 auto;
            display: block; /* Ensure input and button are block-level and centered */
        }

        .btn {
            background-color: #228B22;
            color: white;
            cursor: pointer;
            border: 2px solid white;
        }

        .btn:hover {
            background-color: #196619;
        }

        .logout-btn {
            padding: 10px 20px;
            border: 2px solid white;
            border-radius: 20px;
            background-color: #444;
            color: white;
            cursor: pointer;
            margin: 20px auto; /* Centered and spaced from the top */
            display: inline-block; /* Ensure button is inline-block and centered */
        }

        .logout-btn:hover {
            background-color: grey;
        }

        a {
            display: block;
            margin-top: 10px;
            color: #007bff;
        }

        a:hover {
            text-decoration: underline;
        }

        .camera-pins, .upload-history {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: left;
            color: black;
            width: calc(100% - 40px);
            margin: 0 auto 20px; /* Ensure sections are centered and have space between them */
            max-height: 300px; /* Fixed height for the box */
            overflow-y: auto; /* Enable scrolling */
        }

        .camera-pins ul, .upload-history ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .camera-pins li, .upload-history li {
            padding: 10px;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .camera-pins li:last-child, .upload-history li:last-child {
            border-bottom: none;
        }

        .btn-delete {
            background-color: #dc3545;
        }

        .btn-delete:hover {
            background-color: #c82333;
        }

        /* Modal styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 1001;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
        }

        .modal-content {
            background-color: #323232;
            margin: 15% auto;
            padding: 20px;
            border: 1px solid #888;
            width: 80%;
            max-width: 500px;
            border-radius: 8px;
            color: white;
        }

        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }

        .close:hover {
            color: white;
        }

        .modal input {
            width: 100%;
            padding: 8px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }

        .modal button {
            width: 100%;
            padding: 10px;
            margin-top: 10px;
            background-color: #228B22;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }

        .modal button:hover {
            background-color: #196619;
        }

        .video-modal {
            display: none;
            position: fixed;
            z-index: 1001;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .video-container {
            width: 80%;
            max-width: 800px;
        }

        video {
            width: 100%;
        }

        .close-video {
            position: absolute;
            right: 20px;
            top: 20px;
            color: white;
            font-size: 30px;
            cursor: pointer;
        }
</style> 