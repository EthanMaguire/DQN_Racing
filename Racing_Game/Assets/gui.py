from Racing_Game.Assets import rectangle
from Racing_Game.Assets import tiles
from Racing_Game.Assets import lines
import pygame
import pygame.freetype
from Racing_Game.Assets import collision
import math
import pickle
import os


def button_callback(callback, button, gui):
    if callback == "Place Points":  # Turns on track drawing using splines
        if not gui.PlacePointsActive:
            print("Placing Points Enabled")
            gui.clear_active_tool()
            gui.active_tool_display.text = "Placing Points Enabled"
            gui.PlacePointsActive = True
        elif gui.PlacePointsActive:
            print("Placing Points Disabled")
            gui.active_tool_display.text = ""
            gui.clear_active_tool()
            gui.send_track_to_world()  # Send the current bezier defined track off to the world

    elif callback == "Save Track":  # Saves the current world track to file
        print("Saving current world track to file")
        index = 0
        while 1:
            ext = "track_" + str(index) + str(".obj")
            fn = "F:/PythonProjects_2024/MachineLearningTesting/Racing_Game/Assets/Tracks/" + ext
            if os.path.exists(fn):
                index += 1
            else:
                gui.world.save_track_pickle(fn)
                break

    elif callback == "Add Finish":
        if gui.PlaceFinishActive:
            print("Draw Finish Disabled")
            gui.active_tool_display.text = ""
            gui.clear_active_tool()
        else:
            print("Draw Finish Enabled")
            gui.active_tool_display.text = "Draw Finish Enabled"
            gui.clear_active_tool()
            gui.PlaceFinishActive = True

    elif callback == "Add Learn Gate":
        if gui.PlaceLearnGateActive:
            print("Draw Learn Gate Disabled")
            gui.active_tool_display.text = ""
            gui.clear_active_tool()
        else:
            print("Draw Learn Gate Enabled")
            gui.active_tool_display.text = "Draw Learn Gate Enabled"
            gui.clear_active_tool()
            gui.PlaceLearnGateActive = True

    elif callback == "Load Track":
        gui.world.clear_world()
        gui.world.load_track_pickle(gui.load_track_fn)
        gui.world.load_from_track()
        print("Track loaded from file: " + gui.load_track_fn)

    elif callback == "Clear Track":
        gui.world.clear_world()
        print("World Cleared")

class GUI:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        # Get object for game world
        self.world = None

        # Define the GUI
        self.objects = []
        # Buttons
        self.objects.append(Button(0, 0, 100, 25, "Draw Track", "Place Points"))
        self.objects.append(Button(105, 0, 100, 25, "Save Track", "Save Track"))
        self.objects.append(Button(210, 0, 100, 25, "Add Finish", "Add Finish"))
        self.objects.append(Button(315, 0, 140, 25, "Add Learn Gate", "Add Learn Gate"))
        self.objects.append(Button(460, 0, 100, 25, "Load Track", "Load Track"))
        self.objects.append(Button(565, 0, 100, 25, "Clear Track", "Clear Track"))
        # Display
        self.active_tool_display = Display(0, 30, 150, 25)


        # Drawing Bezier Track
        self.PlacePointsActive = False
        self.points_list = []
        self.temp_track = []

        # Placing specific blocks
        self.PlaceFinishActive = False
        self.PlaceLearnGateActive = False
        self.temp_pt = [0, 0]
        self.temp_flag = False

        # Loading a specific track
        self.load_track_fn = ""

    def draw(self, surface):
        # sort by z to correctly draw overlapping objects
        z_sorted = sorted(self.objects, key=lambda ob: ob.z)
        for obj in z_sorted:
            obj.draw(surface)

        # Draw displays
        self.active_tool_display.draw(surface)

        # Draw bezier being designed
        if self.PlacePointsActive:
            self.gen_temp_track()

            z_sorted = sorted(self.temp_track, key=lambda ob: ob.z)
            for obj in z_sorted:
                obj.draw(surface)

        elif self.PlaceFinishActive:
            if self.temp_flag:  # Draw the current finish hitbox
                mpt = pygame.mouse.get_pos()
                rect = collision.define_rect(mpt, self.temp_pt)
                pygame.draw.rect(surface, (145, 94, 16), rect)

        elif self.PlaceLearnGateActive:
            if self.temp_flag:  # Draw the current finish hitbox
                mpt = pygame.mouse.get_pos()
                pygame.draw.line(surface, (159, 31, 191), self.temp_pt, mpt, width=5)

    def check_click(self, x, y):
        # Check if the cursor is on any buttons
        for obj in self.objects:
            if isinstance(obj, Button):
                if collision.in_rect(obj.rect, x, y):
                    button_callback(obj.callback, obj, self)
                    return

        # No buttons were pressed
        if self.PlacePointsActive:
            if len(self.points_list) % 2 == 1 and len(self.points_list) != 1:
                p0 = self.points_list[-1]
                p2 = pygame.mouse.get_pos()
                if p2[0] == p0[0]:  # avoid divide by zero on double click
                    p2 = [p2[0] + 10, p2[1]]
                dist = collision.distance(p0[0], p0[1], p2[0], p2[1])  # Distance from p0 to the mouse
                p_last = self.points_list[-2]
                last_vector = [p_last[0] - p0[0], p_last[1] - p0[1]]
                dis = math.hypot(*last_vector)
                norm = (-last_vector[0] / dis, -last_vector[1] / dis)
                p2 = [p0[0] + norm[0] * dist, p0[1] + norm[1] * dist]
                p1 = [(p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2]

                self.points_list.append((p1[0], p1[1]))
            else:
                self.points_list.append((x, y))

        elif self.PlaceFinishActive:
            if not self.temp_flag:  # If the starting point hasn't been placed, place it.
                self.temp_flag = True
                self.temp_pt = pygame.mouse.get_pos()
            elif self.temp_flag:  # Send the finish to the world
                mpt = pygame.mouse.get_pos()
                rect = collision.define_rect(mpt, self.temp_pt)
                self.world.add_object(tiles.Finish_Box(rect[0], rect[1], rect[2], rect[3], z=1))
                self.PlaceFinishActive = False
                self.temp_flag = False
                print("Finish Added to Track")

        elif self.PlaceLearnGateActive:
            if not self.temp_flag:  # If the starting point hasn't been placed, place it.
                self.temp_flag = True
                self.temp_pt = pygame.mouse.get_pos()
            elif self.temp_flag:  # Send the finish to the world
                mpt = pygame.mouse.get_pos()
                self.world.add_object(tiles.Learn_Gate(self.temp_pt, mpt, z=1))
                self.temp_flag = False
                print("Learn Gate Added to Track")

    def send_track_to_world(self):
        self.gen_temp_track(final=True)
        for obj in self.temp_track:
            self.world.objects.append(obj)

    def gen_temp_track(self, final=False):
        self.temp_track = []
        if len(self.points_list) > 0:  # At least one point exists
            index = 0
            while 1:
                remaining = len(self.points_list) - index
                if remaining <= 0:
                    break
                elif remaining < 3 and final:
                    break
                elif remaining >= 3:
                    self.temp_track.append(tiles.Pure_Bezier(self.points_list[index], self.points_list[index + 1],
                                                             self.points_list[index + 2], z=-1))
                    index += 2
                    continue
                elif remaining == 2:
                    self.temp_track.append(tiles.Pure_Bezier(self.points_list[index], self.points_list[index + 1],
                                                             pygame.mouse.get_pos(), z=-1))
                    break
                if remaining == 1:
                    if index == 0:  # Placing first point
                        p0 = self.points_list[index]
                        p2 = pygame.mouse.get_pos()
                        if p2[0] == p0[0]:
                            p2 = [p2[0] + 10, p2[1]]
                        p1 = [(p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2]
                    else:  # Placing aligned points for followup beziers
                        p0 = self.points_list[index]
                        p2 = pygame.mouse.get_pos()
                        if p2[0] == p0[0]:
                            p2 = [p2[0] + 10, p2[1]]
                        dist = collision.distance(p0[0], p0[1], p2[0], p2[1])  # Distance from p0 to the mouse
                        p_last = self.points_list[index - 1]
                        last_vector = [p_last[0] - p0[0], p_last[1] - p0[1]]
                        dis = math.hypot(*last_vector)
                        norm = (-last_vector[0] / dis, -last_vector[1] / dis)
                        p2 = [p0[0] + norm[0] * dist, p0[1] + norm[1] * dist]
                        p1 = [(p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2]

                    self.temp_track.append(tiles.Pure_Bezier(p0, p1, p2, z=-1))
                    break

    def clear_active_tool(self):
        self.PlaceFinishActive = False
        self.PlaceLearnGateActive = False
        self.PlacePointsActive = False


class Button:
    def __init__(self, x, y, w, h, text, callback):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.rect = [x, y, w, h]
        self.z = 0
        self.font = pygame.freetype.SysFont("Arial", 20)
        self.text = text
        self.callback = callback

        self.clicked = False
        self.active = False

        self.objects = []
        # Background
        self.objects.append(rectangle.Rectangle(self.x, self.y, self.w, self.h, color=(42, 43, 46), z=-1))
        # Clickable part
        self.objects.append(
            rectangle.Rectangle(self.x + 3, self.y + 3, self.w - 6, self.h - 6, color=(161, 185, 204), z=0))

    def draw(self, surface):
        # sort by z to correctly draw overlapping objects
        z_sorted = sorted(self.objects, key=lambda ob: ob.z)
        for obj in z_sorted:
            obj.draw(surface)

        # Render Font
        self.font.render_to(surface, (self.x + 5, self.y + 5), self.text, size=22)


class Display:
    def __init__(self, x, y, w, h, text=""):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.rect = [x, y, w, h]
        self.z = 0
        self.font = pygame.freetype.SysFont("Arial", 20)
        self.text = text

        self.objects = []
        # Background
        self.objects.append(rectangle.Rectangle(self.x, self.y, self.w, self.h, color=(255, 255, 255), z=-1))

    def draw(self, surface):
        # sort by z to correctly draw overlapping objects
        z_sorted = sorted(self.objects, key=lambda ob: ob.z)
        for obj in z_sorted:
            obj.draw(surface)

        # Render Font
        self.font.render_to(surface, (self.x + 5, self.y + 5), self.text, size=22, fgcolor=(9, 99, 26))
