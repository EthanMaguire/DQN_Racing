import pickle
import time

import pygame
import numpy as np
from Racing_Game.Assets import car_v1
import pygame.gfxdraw as gfx
import math
import random
from Racing_Game.Assets import collision


def calc_bezier(x0, x1, x2, t):
    return (1 - t) ** 2 * x0 + 2 * (1 - t) * t * x1 + t ** 2 * x2


def calc_edge_points(pts, norms, width):
    points = pts.copy()
    up_points = []
    down_points = []
    for i, pt in enumerate(points):
        n = norms[i]  # normalized vector
        p = (-n[1], n[0])  # perpendicular vector
        sp = (p[0] * width / 2, p[1] * width / 2)  # scaled perpendicular vector
        up_points.append(pygame.Vector2(pt[0] + sp[0], pt[1] + sp[1]))
        down_points.append(pygame.Vector2(pt[0] - sp[0], pt[1] - sp[1]))

    return up_points, down_points


def calc_bezier_points(p0, p1, p2, numpoints):
    t_ls = np.linspace(0, 1, num=numpoints, endpoint=True)
    points = []
    for t in t_ls:
        x = calc_bezier(p0[0], p1[0], p2[0], t)
        y = calc_bezier(p0[1], p1[1], p2[1], t)
        points.append(pygame.Vector2(x, y))

    return points


def calc_bezier_norms(points):
    # Get normalized vectors pointing from pt[i] to pt[i + 1]
    norm_vectors = []
    for i, pt in enumerate(points):
        if i == len(points) - 1:
            d = (points[i][0] - points[0][0], points[i][1] - points[0][1])
        else:
            d = (points[i][0] - points[i + 1][0], points[i][1] - points[i + 1][1])
        dis = math.hypot(*d)  # distance between the points
        norm_vectors.append(pygame.Vector2(d[0] / dis, d[1] / dis))  # normalized vector added to list

    return norm_vectors


def calc_blended_vec2s(vec2s, params):
    assert params[0] % 2 == 1, "Blend[0] must be odd number"

    old_vec2s = vec2s.copy()
    blended_vec2s = vec2s.copy()
    # Blend the normalized vectors based on nearby values
    for _ in range(params[1]):
        blended_vec2s = old_vec2s.copy()
        for i, vec in enumerate(blended_vec2s):
            blend_ls = []
            for j in range(params[0]):
                offset = int((j - (params[0] - 1) / 2)) % len(old_vec2s)  # So that index wraps around
                blend_ls.append(old_vec2s[i - offset])

            blended_vec2s[i] = pygame.Vector2(np.mean([val.x for val in blend_ls]), np.mean([val.y for val in blend_ls]))

        old_vec2s = blended_vec2s.copy()

    return blended_vec2s


def calc_spread_vec2s(vec2s, params):
    assert params[0] % 2 == 1, "Blend[0] must be odd number"

    old_vec2s = vec2s.copy()
    spread_vec2s = vec2s.copy()
    # Blend the normalized vectors based on nearby values
    for _ in range(params[1]):
        spread_vec2s = old_vec2s.copy()
        for i, vec in enumerate(old_vec2s):
            blend_ls = []
            for j in range(params[0]):
                offset = int((j - (params[0] - 1) / 2)) % len(old_vec2s)  # So that index wraps around
                blend_ls.append(old_vec2s[i - offset])

            mean_pt = pygame.Vector2(np.mean([val.x for val in blend_ls]), np.mean([val.y for val in blend_ls]))
            diff = pygame.Vector2(old_vec2s[i][0] - mean_pt[0], old_vec2s[i][1] - mean_pt[1]) * 1.0
            spread_vec2s[i] = pygame.Vector2(spread_vec2s[i][0] - diff[0], spread_vec2s[i][1] - diff[1])

        old_vec2s = spread_vec2s.copy()

    return spread_vec2s


class raceGame:
    def __init__(self, screen_bounds, use_model=False, track_id=-1):
        # Track Setup
        self.screen_bounds = screen_bounds
        self.track_id = track_id
        self.track_dir = "F:/PythonProjects_2024/MachineLearningTesting/Racing_Game/Assets/Tracks/"
        if track_id == -1:
            self.track = Track(self.screen_bounds, mk_random=True)

        else:
            self.track = self.load_track()

        # Initialize a car
        self.gameCar = car_v1.CarV1(self.track.start_position, angle=self.track.start_angle, size=0.15, speed=0)
        self.gameCar.controller.use_model = use_model

        # Misc for now
        self.finishFlag = False

    def draw(self, screen):
        # self.track.draw(screen)
        self.gameCar.draw(screen)

        # Debug
        self.get_state()
        for pt in self.debug_pts:
            pygame.draw.circle(screen, (0, 0, 255), pt, 5, 0)

        pygame.display.flip()  # flip() the display to put your work on screen
        screen.fill("black")  # fill the screen with a color to wipe away anything from last frame

    def step(self, dt, draw=False, screen=None, action=None):
        if action is not None:
            action = action.cpu().numpy()[0]  # send act to the cpu then convert to a numpy array

        self.gameCar.controller.update(dt, action=action)  # Send the action to the controller
        self.gameCar.update(dt)  # update the car

        if draw:  # Draw each object in the game world
            # Check events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit

            self.draw(screen)

        obs = self.get_state()
        rew = self.get_reward()
        dead = self.gameCar.collide_flag
        fin = self.finishFlag
        return obs, rew, dead, fin

    def load_track(self):
        return pickle.load(open(self.get_track_fn(), 'rb'))

    # def save_track(self, fn):
    #     with open(fn, 'wb') as f:
    #         pickle.dump(self.track, f)
    #     print("Track saved to file: " + fn)

    def get_track_fn(self):
        return self.track_dir + "Track_" + str(self.track_id) + ".obj"

    def get_reward(self):
        return 0

    def reset_env(self):
        self.gameCar = car_v1.CarV1(self.track.start_position[0], angle=self.track.start_angle, size=0.15, speed=0)

    def get_state(self, points=50):
        start_index = self.check_car_collision()
        # State is the cars current speed, and a vector (dist, dtheta)
        # in reference to the driving direction of the next N points on the track
        state = []
        state.append(self.gameCar.speed / self.gameCar.controller.max_speed)  # Norm speed
        self.debug_pts = []
        for i in range(points):  # append a vector towards the next N points
            index = (i + start_index) % len(self.track.bez_points)
            # Debug
            self.debug_pts.append(self.track.bez_points[index])

    def check_car_collision(self):
        # check if the car has touched any box checkpoints to get a starting point
        start_index = -1
        for i, box in enumerate(self.track.boxes):
            if not self.track.boxes_touched[i]:
                start_index = i
                break

        if start_index == -1:  # car has completed the course
            self.finishFlag = True
        else:  # Car isn't done check if its hitting the next one
            if collision.check_rect_rect(self.gameCar.hitbox, self.track.boxes[start_index]):
                self.track.boxes_touched[start_index] = True
                start_index += 1

        return start_index
        # TODO Check if the car hits the walls (later)


class Track:
    def __init__(self, bounds, mk_random):
        # Track contains 3 lists of points (Vec2s) which are p0, p1
        # p0 are the end points while p1 has the control points
        # p0[0] is used as the final attachment point for the track
        self.p_zeros = []
        self.p_ones = []
        self.track_color = (255, 255, 255)
        self.interpolate_numpoints = 30
        self.num_curves = 10
        self.width = 60
        self.bounds = bounds
        self.draw_points = []  # list of pts describing a polygon width wide that follows the entire track
        self.bez_points = []
        self.bez_norms = []
        self.ups = []
        self.downs = []
        # self.blend_params_points = [11, 7]
        # self.blend_params_points = [7, 50]
        # self.blend_params_points = [1, 1]
        if mk_random:
            t1 = time.perf_counter()
            self.gen_track()
            print("Track generated in " + str(time.perf_counter() - t1) + " seconds")
            self.transform_track()  # This is redundant, but it's not expensive to run in case check_track isn't

        # Car positioning and collision
        self.start_position = self.bez_points[0]
        self.start_angle = math.degrees(math.acos(self.bez_norms[0][0])) + 90
        self.boxes = []
        self.boxes_touched = []
        self.gen_boxes()


    def gen_track(self):
        num_tries = 0
        while 1:
            num_tries += 1

            # self.random_track()
            self.spoke_track()

            # Check if the track overlaps itself
            if self.check_valid_track():  # check that the track is valid
                print("Track generated in: " + str(num_tries) + " attempts")
                return

    def draw(self, screen):
        gfx.filled_polygon(screen, self.draw_points, self.track_color)

        # Debug
        for pt in self.p_zeros:
            pygame.draw.circle(screen, (255, 0, 0), pt, 10)
        for pt in self.p_ones:
            pygame.draw.circle(screen, (0, 255, 0), pt, 10)
        pygame.draw.circle(screen, (0, 0, 255), self.p_ones[-1], 10)
        pygame.draw.circle(screen, (0, 125, 125), self.p_zeros[0], 10)

        # for pt in self.bez_points:
        #     pygame.draw.circle(screen, (255, 0, 0), pt, 10)
        #
        # for pt in self.draw_points:
        #     pygame.draw.circle(screen, (0, 255, 0), pt, 3)

    def compute_interpolate_points(self, num_points):
        # Compute both the outside and inside edge for each curve in the bezier track and create two lists
        self.bez_points = []
        first = True
        for i in range(len(self.p_zeros) - 1):  # Get all points and their norms (direction to next point)
            if first:  # do final set connecting the final point to the start
                first = False
                pts = calc_bezier_points(self.p_zeros[-1], self.p_ones[-1], self.p_zeros[i], num_points)
                self.bez_points += pts[1:]  # First point is redundant for closed bezier loops after the start

            pts = calc_bezier_points(self.p_zeros[i], self.p_ones[i], self.p_zeros[i + 1], num_points)
            self.bez_points += pts[1:]  # First point is redundant for closed bezier loops after the start

        # Blend the points to smooth the curve
        # self.bez_points = calc_blended_vec2s(self.bez_points, self.blend_params_points)
        # self.bez_norms = calc_bezier_norms(self.bez_points)
        # self.bez_norms = calc_blended_vec2s(self.bez_norms, self.blend_params_norms)
        # self.bez_points = calc_spread_vec2s(self.bez_points, self.blend_params_points)
        self.bez_norms = calc_bezier_norms(self.bez_points)

        # Compute the ups / downs using the blended norms
        ups, downs = calc_edge_points(self.bez_points, self.bez_norms, self.width)
        self.draw_points = ups + [ups[0]] + [downs[0]] + downs[::-1]  # go around the using ups in reverse along downs
        self.ups = ups
        self.downs = downs

    def check_valid_track(self):
        self.compute_interpolate_points(self.interpolate_numpoints)
        if collision.check_poly_lines(self.downs, self.downs, closed_poly=False, self_check=True, self_skip=1)[0]:
            return False
        if collision.check_poly_lines(self.ups, self.ups, closed_poly=False, self_check=True)[0]:
            return False
        # if collision.check_poly_lines(ups, downs, closed_poly=False, self_check=False)[0]:
        #     return False

        # Check if the final control point is on the screen
        elif not collision.check_pt_in_segment(self.p_ones[-1], [pygame.Vector2(), self.bounds]):
            return False

        return True

    def random_track(self):  # Almost completely stochastic point placement, some logic for forcing continuity
        # Generate random sets of points
        r1x, r1y, c1x, c1y = 0, 0, 0, 0
        self.p_zeros = []
        self.p_ones = []
        for i in range(self.num_curves):
            # Track points are bounded by the screen edge - the width
            r1x = random.uniform(self.width, self.bounds[0] - self.width)
            r1y = random.uniform(self.width, self.bounds[1] - self.width)
            if i == 0:  # gen the first control point randomly
                c1x = random.uniform(self.width, self.bounds[0] - self.width)
                c1y = random.uniform(self.width, self.bounds[0] - self.width)

            if i == self.num_curves - 1:
                # r[-1] needs to be between r[-2], r[0] or else there is an inversion
                left_bound = min(self.p_zeros[0].x, self.p_zeros[i - 1].x)
                r1x = random.uniform(min(self.p_zeros[0].x, self.p_zeros[i - 1].x),
                                     max(self.p_zeros[0].x, self.p_zeros[i - 1].x))
                r1y = random.uniform(min(self.p_zeros[0].y, self.p_zeros[i - 1].y),
                                     max(self.p_zeros[0].y, self.p_zeros[i - 1].y))
                # last control point is deterministic as the intersection of a line extending from c[0] --> r[0]
                # and line c[old] --> r[new]
                c1x, c1y = collision.get_line_line_intersect([r1x, r1y], [c1x, c1y],
                                                             self.p_zeros[0], self.p_ones[0])

            else:  # now the control point needs to follow the line c[old]-->r[new]
                dist = random.uniform(50, 200)  # Pick a distance for the new control point
                mag = math.sqrt((r1x - c1x) ** 2 + (r1y - c1y) ** 2)  # Get the distance from the old control point
                n = pygame.Vector2((c1x - r1x) / mag, (c1y - r1y) / mag)  # get the unit vector c[old] -->r[new]
                c1x = r1x - n.x * dist
                c1y = r1y - n.y * dist

            self.p_zeros.append(pygame.Vector2(r1x, r1y))
            self.p_ones.append(pygame.Vector2(c1x, c1y))

    def spoke_track(self):
        # random.seed(371)
        self.p_zeros = []
        self.p_ones = []
        shift_factor = .8
        shift_fixed = 10
        theta_lower = 0.3
        theta_upper = 0.35
        # places endpoints around a circle evenly spaced by angle
        # randomly chooses a radius for them along the spokes
        # do math in polar cords
        dtheta = 2 * math.pi / self.num_curves
        zeros = []
        ones = []
        for i in range(self.num_curves):  # randomly place the endpoints along spokes
            radi = random.uniform(self.width / 4, (self.bounds[1] / 2) - self.width * 2.5)
            zeros.append(pygame.Vector2(radi, i * dtheta))

            # iteratively generate the control points
            if i == 0:  # first one is random r and theta within bounds
                c1r = random.uniform(self.width / 4, (self.bounds[0] / 2) - self.width * 2.5)
                c1t = random.uniform(i * dtheta + dtheta * theta_lower, i * dtheta + dtheta * theta_upper)
                ones.append(pygame.Vector2(c1r, c1t))
            else:  # for subsequent ones choose only theta
                for j in range(10):
                    c1t = random.uniform(i * dtheta + dtheta * theta_lower, i * dtheta + dtheta * theta_upper)
                    # c1r is the intersection between lines ones[i - 1] -- >zeros[i] and origin-->ct1
                    cart_l1 = [collision.pol2cart(ones[i - 1]), collision.pol2cart(zeros[i])]
                    cart_l2 = [pygame.Vector2(), collision.pol2cart(pygame.Vector2(1, c1t))]
                    cart_intercept = collision.get_line_line_intersect(cart_l1[0], cart_l1[1], cart_l2[0], cart_l2[1])
                    cart_intercept_pol = collision.cart2pol(cart_intercept)
                    if cart_intercept_pol[1] < 0:
                        cart_intercept_pol[1] += math.pi * 2
                    if not abs(cart_intercept_pol[1] - c1t) > 0.01:
                        break
                    if j == 9:
                        print(ones[i - 1], zeros[i])
                        self.spoke_track()  # Restart if the intercept is on the other side of the circle
                    zeros[i].x *= shift_factor  # Shift z towards the center to fix intercept missing error
                    zeros[i].x -= shift_fixed
                    if abs(zeros[i].x) < self.width / 4:
                        self.spoke_track()  # Restart if the intercept is on the other side of the circle

                ones.append(cart_intercept_pol)

            if i == self.num_curves - 1:
                # on final iteration move the start point to be collinear with c1[-1] --> c1[0]
                cart_l1 = [collision.pol2cart(ones[-1]), collision.pol2cart(ones[0])]
                cart_l2 = [pygame.Vector2(), collision.pol2cart(pygame.Vector2(1, 0))]
                cart_intercept = collision.get_line_line_intersect(cart_l1[0], cart_l1[1], cart_l2[0], cart_l2[1])
                zeros[0] = collision.cart2pol(cart_intercept)

        for i in range(self.num_curves):
            # convert to cart
            self.p_zeros.append(collision.pol2cart(zeros[i]))
            self.p_ones.append(collision.pol2cart(ones[i]))
            # Shift by center of the track
            self.p_zeros[i] = pygame.Vector2(self.p_zeros[i].x + self.bounds.x / 2,
                                             self.p_zeros[i].y + self.bounds.y / 2)
            self.p_ones[i] = pygame.Vector2(self.p_ones[i].x + self.bounds.x / 2,
                                            self.p_ones[i].y + self.bounds.y / 2)

    def transform_track(self):
        corners = collision.get_corners(self.bez_points)
        dx = self.bounds[0] / 2 - min(corners[0][0], self.bounds[0] - corners[1][0]) + self.width / 2
        dy = self.bounds[1] / 2 - min(corners[0][1], self.bounds[1] - corners[1][1]) + self.width / 2
        x_scale = (self.bounds[0] / 2) / dx
        y_scale = (self.bounds[1] / 2) / dy
        for i in range(len(self.p_zeros)):
            self.p_zeros[i].x -= self.bounds.x / 2  # Shift to zero
            self.p_zeros[i].x *= x_scale  # Scale
            self.p_zeros[i].x += self.bounds.x / 2  # Shift back

            self.p_zeros[i].y -= self.bounds.y / 2  # Shift to zero
            self.p_zeros[i].y *= y_scale  # Scale
            self.p_zeros[i].y += self.bounds.y / 2  # Shift back

            self.p_ones[i].x -= self.bounds.x / 2  # Shift to zero
            self.p_ones[i].x *= x_scale  # Scale
            self.p_ones[i].x += self.bounds.x / 2  # Shift back

            self.p_ones[i].y -= self.bounds.y / 2  # Shift to zero
            self.p_ones[i].y *= y_scale  # Scale
            self.p_ones[i].y += self.bounds.y / 2  # Shift back

        self.compute_interpolate_points(self.interpolate_numpoints)

    def gen_boxes(self):  # using up/down pairs generate pairs of corner values
        self.boxes = []
        self.boxes_touched = []
        for i in range(len(self.ups) - 1):
            if i == 0:
                self.boxes.append(
                    collision.get_corners([self.ups[i - 1], self.ups[i], self.downs[i - 1], self.downs[i]]))
            else:
                self.boxes.append(
                    collision.get_corners([self.ups[i], self.ups[i + 1], self.downs[i], self.downs[i + 1]]))

            self.boxes_touched.append(False)
