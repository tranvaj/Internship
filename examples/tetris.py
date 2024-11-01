import numpy as np
import cv2
from config import TetrioConfig, TetrisConfig
import easyocr



class TetrisGame():
    def __init__(self, config: TetrisConfig):
        self.config = config

    def shrink_sides(self, box, left_percentage, right_percentage):
        # Calculate the center of the box
        center = np.mean(box, axis=0)

        # Calculate the vectors from the center to the points
        vectors = box - center

        # Adjust the x-coordinates of the vectors
        for vector in vectors:
            if vector[0] < 0:
                vector[0] *= left_percentage
            else:
                vector[0] *= right_percentage

        # Calculate the new box points
        new_box = center + vectors

        return new_box.astype(int)

    def shrink_vertically(self, box, top_percentage, bottom_percentage):
        # Calculate the center of the box
        center = np.mean(box, axis=0)

        # Calculate the vectors from the center to the points
        vectors = box - center

        # Adjust the y-coordinates of the vectors
        for vector in vectors:
            if vector[1] < 0:
                vector[1] *= top_percentage
            else:
                vector[1] *= bottom_percentage

        # Calculate the new box points
        new_box = center + vectors

        return new_box.astype(int)
    


    def detect_board_state2(self, box_board, rgb_image, debug=False):
        box_board_cv2 = np.array(box_board, dtype=np.float32)

        # Calculate the bounding rectangle
        x, y, w, h = cv2.boundingRect(box_board_cv2)

        board = rgb_image[y:y+h, x:x+w]

        board_height = self.config.board_height
        board_width = self.config.board_width

        # Improved cell size calculation
        cell_heights = [h // board_height + (i < h % board_height) for i in range(board_height)]
        cell_widths = [w // board_width + (i < w % board_width) for i in range(board_width)]

        board_state = np.zeros((board_height, board_width), dtype=str)
        
        # Create a copy of the board for drawing the debug grid
        debug_board = board.copy() if debug else None

        current_y = 0
        for i in range(board_height):
            current_x = 0
            for j in range(board_width):
                cell = board[current_y:current_y+cell_heights[i], current_x:current_x+cell_widths[j]]
                
                # Draw the cell boundaries for debug purposes
                if debug:
                    cv2.rectangle(debug_board, (current_x, current_y), (current_x + cell_widths[j], current_y + cell_heights[i]), (0, 255, 0), 1)
                
                current_x += cell_widths[j]

                if np.isnan(cell).all() or cell.size == 0:
                    continue

                # Improved color detection logic
                avg_color = np.mean(cell.reshape(-1, 3), axis=0).astype(int)
                min_diff = float('inf')
                closest_piece = None
                for piece, color in self.config.colors.items():
                    diff = np.linalg.norm(avg_color - color)
                    if diff < min_diff:
                        min_diff = diff
                        closest_piece = piece
                if min_diff < self.config.color_diff_board_threshold:
                    board_state[i, j] = closest_piece
                    if debug:
                        print(min_diff, closest_piece)
                else:
                    board_state[i, j] = '-'
            current_y += cell_heights[i]

        # Display the debug board with grid
        if debug:
            cv2.imshow('Debug Board with Grid', debug_board)
            cv2.waitKey(0)  # Wait for a key press
            cv2.destroyAllWindows()  # Close the window

        return board_state

    
    def adjust_box_to_aspect_ratio(self, box, aspect_ratio):
        x, y, w, h = cv2.boundingRect(box)

        # Calculate the desired width and height
        if w >= h * aspect_ratio:
            if aspect_ratio == 1/2:
                #make width divisible by 10 (round up to nearest 10) for cell size
                w = (w + 5) // 10 * 10
            desired_width = w
            desired_height = w / aspect_ratio
        else:
            desired_height = h
            desired_width = h * aspect_ratio

        # Calculate the new x and y coordinates to center the new bounding rectangle
        new_x = x + w/2 - desired_width/2
        new_y = y + h/2 - desired_height/2

        # Create the new box
        new_box = np.array([
            [round(new_x), round(new_y)],
            [round(new_x + desired_width), round(new_y)],
            [round(new_x + desired_width), round(new_y + desired_height)],
            [round(new_x), round(new_y + desired_height)]
        ], dtype=np.intp)

        return new_box
    
    def adjust_box(self, image, box):
        # Get the minimum and maximum x and y coordinates
        min_x, min_y = np.min(box, axis=0)
        max_x, max_y = np.max(box, axis=0)

        # Adjust the coordinates if they go outside of the image dimensions
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(max_x, image.shape[1])
        max_y = min(max_y, image.shape[0])

        # Create a new box that is within the image dimensions
        return np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]], dtype=np.intp)

    def detect_game_state(self, image, debug=False):
        # Preprocess the image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, self.config.threshold, 255, cv2.THRESH_BINARY)    # Find contours of the Tetris board
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            if self.config.lock_contour and self.config.board_contour is not None:
                board_contour = self.config.board_contour
            else:
                board_contour = max(contours, key=lambda contour: cv2.boundingRect(contour)[3])
                self.config.board_contour = board_contour
        else:
            print("No contours found")
            return
        
        # Find the rotated rectangle
        rect = cv2.minAreaRect(board_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        box_board = self.shrink_sides(box, self.config.board_left, self.config.board_right)
        box_board = self.shrink_vertically(box_board, self.config.board_vert_top, self.config.board_vert_bottom)

        box_next = self.shrink_sides(box, self.config.next_left, self.config.next_right)
        box_next = self.shrink_vertically(box_next, self.config.next_vert_top, self.config.next_vert_bottom)

        box_held = self.shrink_sides(box, self.config.held_left, self.config.held_right)
        box_held = self.shrink_vertically(box_held, self.config.held_vert_top, self.config.held_vert_bottom)

        box_garbage = self.shrink_sides(box, self.config.garbage_left, self.config.garbage_right)

        # Adjust the boxes
        box_board = self.adjust_box(image, box_board)
        box_next = self.adjust_box(image, box_next)
        box_held = self.adjust_box(image, box_held)
        box_garbage = self.adjust_box(image, box_garbage)

        if debug:
            cv2.drawContours(image, [box_board], 0, (0, 255, 0), 1)
            cv2.drawContours(image, [box], 0, (233, 123, 123), 2)
            cv2.drawContours(image, [box_next], 0, (233, 123, 123), 2)
            cv2.drawContours(image, [box_held], 0, (77, 122, 154), 2)
            cv2.drawContours(image, [box_garbage], 0, (0, 0, 231), 2)

        held_piece = self.detect_held_piece(box_held, rgb_image)
        next_pieces = self.detect_next_pieces(box_next, rgb_image)
        board_state = self.detect_board_state2(box_board, rgb_image, debug=debug)
        garbage_lines = self.detect_garbage(box_garbage, rgb_image)

        return (board_state, held_piece, next_pieces, garbage_lines)

    def detect_next_pieces(self, box_next, rgb_image):
        # Extract the region of the image corresponding to the next pieces
        x, y, w, h = cv2.boundingRect(box_next)
        next_region = rgb_image[y:y+h, x:x+w]

        # Divide the region into cells
        cell_height, cell_width = h // self.config.next_pieces, w
        next_pieces = []

        # Detect the color of each cell
        for i in range(self.config.next_pieces):
            cell = next_region[i*cell_height:(i+1)*cell_height, :]
            
            # Resize the cell image to a larger size
            resized_cell = cv2.resize(cell, (cell_width*2, cell_height*2))
            
            # Apply Gaussian blur to reduce noise
            blurred_cell = cv2.GaussianBlur(resized_cell, (5, 5), 0)
            
            # Convert the cell image to grayscale
            gray_cell = cv2.cvtColor(blurred_cell, cv2.COLOR_RGB2GRAY)
            
            # Apply thresholding to create a binary image
            _, thresh_cell = cv2.threshold(gray_cell, 50, 255, cv2.THRESH_BINARY)
            
            # Find contours in the binary image
            contours, _ = cv2.findContours(thresh_cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # Find the largest contour which is expected to be the tetromino
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Create a mask where the contour is
                mask = np.zeros_like(gray_cell)
                cv2.drawContours(mask, [largest_contour], -1, 255, -1) # -1 to fill the contour
                
                # Calculate the average color of the ROI using the mask
                avg_color = cv2.mean(resized_cell, mask=mask)[:3]  # Only take the first three channels
                
                min_diff = float('inf')
                closest_piece = None
                
                for piece, color in self.config.colors.items():
                    if piece == 'G' or piece == 'GM' or piece == 'GM2':  # Ignore piece G
                        continue
                    diff = np.linalg.norm(np.array(avg_color) - np.array(color))
                    if diff < min_diff:
                        min_diff = diff
                        closest_piece = piece
                
                if min_diff < self.config.color_diff_next_threshold:
                    #print(min_diff, closest_piece)
                    next_pieces.append(closest_piece)

        return next_pieces

    def detect_held_piece(self, box_held, rgb_image):
        # Extract the region of the image corresponding to the held piece
        x, y, w, h = cv2.boundingRect(box_held)

        # Check if box_held is valid
        if x >= 0 and y >= 0 and x + w <= rgb_image.shape[1] and y + h <= rgb_image.shape[0]:
            held_region = rgb_image[y:y+h, x:x+w]
        else:
            print("Invalid box_held")
            return

        cell_height, cell_width, _ = held_region.shape
        cell = held_region

        # Resize the cell image to a larger size
        resized_cell = cv2.resize(cell, (cell_width*2, cell_height*2))
        
        # Apply Gaussian blur to reduce noise
        blurred_cell = cv2.GaussianBlur(resized_cell, (5, 5), 0)
        
        # Convert the cell image to grayscale
        gray_cell = cv2.cvtColor(blurred_cell, cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding to create a binary image
        _, thresh_cell = cv2.threshold(gray_cell, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh_cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        if len(contours) > 0:
            # Find the largest contour which is expected to be the held piece
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create a mask where the contour is
            mask = np.zeros_like(gray_cell)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1) # -1 to fill the contour
            
            # Calculate the average color of the ROI using the mask
            avg_color = cv2.mean(resized_cell, mask=mask)[:3]  # Only take the first three channels
            
            min_diff = float('inf')
            closest_piece = None
            
            for piece, color in self.config.colors.items():
                if piece == 'G':  # Ignore piece G
                    continue
                diff = np.linalg.norm(np.array(avg_color) - np.array(color))
                if diff < min_diff:
                    min_diff = diff
                    closest_piece = piece

            if min_diff < self.config.color_diff_hold_threshold:
                return closest_piece
        return None

    def detect_garbage(self, box_garbage, rgb_image):
        # Extract the region of the image corresponding to the garbage area
        x, y, w, h = cv2.boundingRect(box_garbage)
        garbage_region = rgb_image[y:y+h, x:x+w]

        # The height of one row in the garbage area
        row_height = h // self.config.board_height

        # Initialize the count of garbage lines
        garbage_lines = 0

        # Check each horizontal line in the garbage region for the presence of the garbage color
        for i in range(self.config.board_height):
            # Extract one row of pixels from the garbage region
            row = garbage_region[i * row_height: (i + 1) * row_height, :]

            if np.isnan(row).all() or row.size == 0:
                continue
            # Calculate the average color of this row
            avg_color = tuple(np.average(row, axis=(0, 1)).astype(int))
            

            # Calculate the difference between the average color and the garbage color
            garbage_color = self.config.colors["GM"]  # The color defined for garbage blocks
            diff = np.linalg.norm(np.array(avg_color) - np.array(garbage_color))
            black = (0, 0, 0)
            diff_black = np.linalg.norm(np.array(avg_color) - np.array(black))


            # If the difference is below a certain threshold, increment the garbage line count
            if diff_black > self.config.color_diff_garbage_threshold:
                garbage_lines += 1

        # Return the total count of garbage lines detected
        return garbage_lines


    def get_remaining_pieces(self, current_piece, next_pieces):
        # Define the order of pieces used by TETR.IO
        tetromino_order = ["Z", "L", "O", "S", "I", "J", "T"]

        # Create a copy of the tetromino order
        remaining_pieces = tetromino_order.copy()

        # Remove the current piece from the remaining pieces
        if current_piece in remaining_pieces:
            remaining_pieces.remove(current_piece)

        # Remove the next pieces from the remaining pieces
        for piece in next_pieces:
            if piece in remaining_pieces:
                remaining_pieces.remove(piece)

        return remaining_pieces

