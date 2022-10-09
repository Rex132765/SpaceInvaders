import pygame as pg
import time
import random
from os.path import dirname, exists
import numpy as np
from math import atan2,sin,cos,degrees,copysign

GAME_RESOURCES = {
    "img": {
        "background": "resources/img/background.png",
        "icon": "resources/img/ufo.png",
        "player": "resources/img/player.png",
        "enemy": "resources/img/enemy.png",
        "bullet": "resources/img/bullet.png"
    },
    "sound": {
    }
}

GAME_SETTINGS = {
    "max_enemies" : 10,
    "max_bullets" : 20,
    "enemy_delay" : [300,3000],
    "enemy_speed_h" : [0.3,2],
    "enemy_speed_v" : [0.2,0.8],
    "enemy_cooldown" : [900,2000],
    "enemy_health" : [100,250],
    "enemy_damage" : 100,
    "bullet_damage" : 50,
    "bullet_speed" : 1.5,
    "player_health" : 1000,
    "player_shield" : 500,
    "player_acceleration" : 0.5,
    "player_top_speed" : 5,
    "player_deceleration" : 1.0,
    "player_cooldown" : 300,
    "player_bullet_speed_mul" : 5,
    "player_bullet_damage_mul" : 2,
    "shield_regen"  : 0.5,
    "shield_regen_delay" : 1000,
    "shield_emiter_count" : 16,
    "shield_radius" : 80,
    "shield_base_color_add" : 0,
    "shield_base_color_mul" : 2,
    "shield_const_shield_repl" : 0.005 / 2,
    "shield_transfer_rate" : 0.09 / 2,
}

APP_CONFIG = {
    "title": "Space Invaders",
    "screen": [800,600],
    "fps": 120
}

DEFAULT_DELTA = 9

class Shield:
    def __init__(self, emiter_count, shield_radius, base_color_add, base_color_mul, const_shield_repl, transfer_rate):
        self.emiter_count = emiter_count
        self.shield_radius = shield_radius
        self.base_color_add = base_color_add
        self.base_color_mul = base_color_mul
        self.const_shield_repl = const_shield_repl
        self.transfer_rate = transfer_rate

        self.emiter_energy = np.ones(emiter_count,dtype = np.double)

        emiter_angles_rad = np.linspace(0,2*np.pi * (1-1/emiter_count),emiter_count)
        self.emiter_centers = np.zeros((emiter_count,2),dtype=np.double)

        self.x, self.y = np.meshgrid(np.arange(0,shield_radius * 2, 1, dtype=np.double),np.arange(0,shield_radius * 2, 1, dtype=np.double))
        self.x_diff, self.y_diff = np.meshgrid(np.zeros(shield_radius * 2, dtype=np.double),np.zeros(shield_radius * 2, dtype=np.double))
        self.per_emiter_shield_distance_image = np.zeros((shield_radius * 2,shield_radius * 2),dtype=np.double)

        self.bugged_color = True

        self.shield_image = np.zeros((shield_radius * 2,shield_radius * 2),dtype=np.double)
        self.shield_surf  = pg.Surface((self.shield_radius*2,self.shield_radius*2))
        self.shield_surf.set_alpha(128)

        for i in range(emiter_count):
            self.emiter_centers[i][0] = shield_radius + (shield_radius - 5) * np.cos(emiter_angles_rad[i])
            self.emiter_centers[i][1] = shield_radius + (shield_radius - 5) * np.sin(emiter_angles_rad[i])

        x_dist = self.emiter_centers[0][0] - self.emiter_centers[1][0]
        y_dist = self.emiter_centers[0][1] - self.emiter_centers[1][1]
        self.emiter_spacing = np.sqrt( x_dist * x_dist + y_dist * y_dist )
        self.emiter_col_bbox_half = np.ceil(self.emiter_spacing / 2)

        self.one_constant_flat = np.ones((shield_radius * 2,shield_radius * 2),dtype=np.double)

        self.shield_image_mask = np.zeros((shield_radius * 2,shield_radius * 2))

        shy,shx = np.ogrid[-shield_radius:shield_radius, -shield_radius:shield_radius]
        shmask = shx*shx + shy*shy <= (shield_radius - 5)*(shield_radius - 5)
        self.shield_image_mask[shmask] = 1

    def change_mode(self):
        if self.bugged_color:
            self.bugged_color = False
            self.base_color_mul = 8
            self.base_color_add = 10
            return
        self.bugged_color = True
        self.base_color_mul = GAME_SETTINGS["shield_base_color_mul"]
        self.base_color_add = GAME_SETTINGS["shield_base_color_add"]

    def calc_shield_image(self):
        self.shield_image.fill(0)
        #self.shield_image_rgb.fill(0)
        for i in range(self.emiter_count):
            np.copyto(self.x_diff,self.x)
            np.copyto(self.y_diff,self.y)

            self.x_diff -= self.emiter_centers[i][0]
            self.y_diff -= self.emiter_centers[i][1]
            self.x_diff *= self.x_diff
            self.y_diff *= self.y_diff

            np.sqrt(np.sqrt( self.x_diff + self.y_diff ), out=self.per_emiter_shield_distance_image)
            self.per_emiter_shield_distance_image += np.max(self.per_emiter_shield_distance_image)/4

            np.divide(self.one_constant_flat,self.per_emiter_shield_distance_image,out = self.per_emiter_shield_distance_image)
            
            np.maximum(self.shield_image, 255 * self.emiter_energy[i] * self.per_emiter_shield_distance_image * self.base_color_mul + self.base_color_add, out = self.shield_image)

        np.multiply(self.shield_image,self.shield_image_mask,out = self.shield_image)

    def get_neighbour(self,start, end, index, left = True):
        if left:
            if index == start:
                return end - 1
            return index - 1
        if index == end - 1:
            return start
        return index + 1

    def update_shield(self, display, x, y):
        self.emiter_energy -= self.const_shield_repl
        transfer = 0

        for i in range(0,self.emiter_count):

            left = self.get_neighbour(0,self.emiter_count,i)
            right = self.get_neighbour(0,self.emiter_count,i,False)

            if self.emiter_energy[i] > 0:
                transfer = self.emiter_energy[i] * self.transfer_rate

                if np.isclose(self.emiter_energy[left],self.emiter_energy[right]):
                    if (self.emiter_energy[left] < self.emiter_energy[i]):
                        self.emiter_energy[i] = self.emiter_energy[i] - transfer
                        self.emiter_energy[left] = self.emiter_energy[left] + transfer/2
                        self.emiter_energy[right] = self.emiter_energy[right] + transfer/2
                elif self.emiter_energy[left] < self.emiter_energy[right]:
                    if (self.emiter_energy[left] < self.emiter_energy[i]):
                        self.emiter_energy[i] = self.emiter_energy[i] - transfer
                        self.emiter_energy[left] = self.emiter_energy[left] + transfer
                else:
                    if (self.emiter_energy[right] < self.emiter_energy[i]):
                        self.emiter_energy[i] = self.emiter_energy[i] - transfer
                        self.emiter_energy[right] = self.emiter_energy[right] + transfer

        np.clip(self.emiter_energy,0,1,out = self.emiter_energy)

        self.calc_shield_image()

        if self.bugged_color:
            surf = pg.transform.flip(pg.transform.rotate(pg.surfarray.make_surface(self.shield_image.astype(np.uint8)),90),False,True)
            surf.set_alpha(128)
            display.blit(surf,(x - self.shield_radius,y - self.shield_radius))
        else:
            pg.surfarray.blit_array(self.shield_surf,self.shield_image.astype(np.int64))
            display.blit(pg.transform.flip(pg.transform.rotate(self.shield_surf,90),False,True),(x - self.shield_radius,y - self.shield_radius))

        
class GameEngine:
    def __init__(self, bounds):
        self.bounds = bounds
        # x y xspeed yspeed cooldown cooldown_max health maxhealth
        self.current_spawn_delay = 0
        self.current_enemy_count = 0
        self.enemies = np.zeros((GAME_SETTINGS["max_enemies"],8), dtype = np.double)

        self.enemy_bbox = None
        self.enemy_bbox_2 = None
        self.enemy_img = None

        self.enemy_speed_multiplier = 1
        self.enemy_speed_multiplier_coeff = 1/300000

        # x y exists rot xspeed yspeed
        self.e_bullet_count = 0
        self.e_bullets = np.zeros((GAME_SETTINGS["max_bullets"],6), dtype = np.double)
        self.p_bullet_count = 0
        self.p_bullets = np.zeros((GAME_SETTINGS["max_bullets"],6), dtype = np.double)
        self.bullet_bbox = None
        self.bullet_img = None    

        self.player_x = 1
        self.player_y = 1

        self.player_xvel = 0

        self.player_attack_delay = 0
        self.player_do_attack = False
        self.mov_left = False
        self.mov_right = False

        self.player_bbox = None
        self.player_bbox_2 = None
        self.player_img = None

        self.player_health = GAME_SETTINGS["player_health"]
        self.player_shield = 0
        self.shield_regen_delay = 0

        self.shield = Shield(GAME_SETTINGS["shield_emiter_count"],GAME_SETTINGS["shield_radius"],GAME_SETTINGS["shield_base_color_add"],GAME_SETTINGS["shield_base_color_mul"],GAME_SETTINGS["shield_const_shield_repl"],GAME_SETTINGS["shield_transfer_rate"])
        self.shield_charge_seq = False
        
        self.hud_x = (self.bounds[0] + self.bounds[2])/8
        self.hud_width = self.bounds[2] - (self.bounds[0] + self.bounds[2])/4
        self.hud_height = (self.bounds[1] + self.bounds[3])/32
        self.hud_health_y = self.bounds[3] - (self.bounds[1] + self.bounds[3])/20
        self.hud_shield_y = self.bounds[3] - (self.bounds[1] + self.bounds[3])/10 - (self.bounds[1] + self.bounds[3])/64



    def load_resources(self, relative, dir):
        if relative:
            self.enemy_img = pg.image.load(GAME_RESOURCES["img"]["enemy"])
            self.player_img = pg.image.load(GAME_RESOURCES["img"]["player"])
            self.bullet_img = pg.image.load(GAME_RESOURCES["img"]["bullet"])
        else:
            self.enemy_img = pg.image.load(f'{dir}/{GAME_RESOURCES["img"]["enemy"]}')
            self.player_img = pg.image.load(f'{dir}/{GAME_RESOURCES["img"]["player"]}')
            self.bullet_img = pg.image.load(f'{dir}/{GAME_RESOURCES["img"]["bullet"]}')

        self.enemy_bbox = [self.enemy_img.get_width(),self.enemy_img.get_height()]
        self.enemy_bbox_2 = [self.enemy_img.get_width()/2,self.enemy_img.get_height()/2]
        self.player_bbox = [self.player_img.get_width(),self.player_img.get_height()]
        self.player_bbox_2 = [self.player_img.get_width()/2,self.player_img.get_height()/2]
        self.player_x = self.bounds[2] / 2 - self.player_bbox_2[0]
        self.player_y = self.bounds[3] - (self.bounds[1] + self.bounds[3])/16 * 3 - (self.bounds[1] + self.bounds[3])/20
        self.bullet_bbox = [self.bullet_img.get_width(),self.bullet_img.get_height()]
        self.bullet_bbox_2 = [self.bullet_img.get_width()/2,self.bullet_img.get_height()/2]

    def reset(self):
        self.current_spawn_delay = 0
        self.current_enemy_count = 0
        self.enemies.fill(-1)

        self.enemy_speed_multiplier = 1

        self.e_bullet_count = 0
        self.e_bullets.fill(-1)
        self.p_bullet_count = 0
        self.p_bullets.fill(-1)

        self.player_x = self.bounds[2] / 2 - self.player_bbox_2[0]
        self.player_y = self.bounds[3] - (self.bounds[1] + self.bounds[3])/16 * 3 - (self.bounds[1] + self.bounds[3])/20

        self.player_health = GAME_SETTINGS["player_health"]
        self.player_shield = 0
        self.shield_charge_seq = False
        self.shield.emiter_energy.fill(1)

    def collide_enemy_bullet_player(self,x,y,w,h):
        if abs(self.player_x - x) <= w + self.player_bbox_2[0] and abs(self.player_y - y) <= h + self.player_bbox_2[1]:
            return True
        return False

    def collide_player_bullet_enemy(self,x,y,w,h):
        for i in range(GAME_SETTINGS["max_enemies"]):
            if self.enemies[i][6] > 0:
                if abs(self.enemies[i][0] - x) <= w + self.enemy_bbox_2[0] and abs(self.enemies[i][1] - y) <= h + self.enemy_bbox_2[1]:
                    self.enemies[i][6] -= GAME_SETTINGS["bullet_damage"] * GAME_SETTINGS["player_bullet_damage_mul"]
                    if self.enemies[i][6] < 0:
                        self.enemies[i][6] = -1
                        self.current_enemy_count -=1
                    return True
        return False

    def collide_enemy_bullet_shield(self,x,y,w,h):
        shield_ptr = self.shield.emiter_centers
        shield_spacing = self.shield.emiter_spacing/2
        shield_radius = self.shield.shield_radius
        for i in range(self.shield.emiter_count):
            if abs(self.player_x + self.player_bbox_2[0] + shield_ptr[i][0] - x - w/2 - shield_radius) <= w + shield_spacing and abs(self.player_y + self.player_bbox_2[1] + shield_ptr[i][1] - y - h/2 - shield_radius) <= h + shield_spacing:
                self.shield.emiter_energy[i] += 1
                if (self.shield.emiter_energy[i] > 1):
                    self.shield.emiter_energy[i] = 1
                return True
        return False

    def collide_enemy_player(self,x,y):
        if abs(self.player_x - x) <= self.enemy_bbox_2[0] + self.player_bbox_2[0] and abs(self.player_y - y) <= self.enemy_bbox_2[1] + self.player_bbox_2[1]:
            return True
        return False

    def spawn_enemy(self):
        self.current_enemy_count += 1
        index = np.where(self.enemies[:,6] <= 0)[0][0]

        self.enemies[index][0] = random.randint(self.bounds[0]+self.enemy_bbox[0],self.bounds[2] - self.enemy_bbox[0])
        self.enemies[index][1] = random.randint(self.bounds[1]+self.enemy_bbox[1],int(self.bounds[3]/4))
        self.enemies[index][2] = (-1 if (random.random()< 0.5) else 1) * random.uniform(GAME_SETTINGS["enemy_speed_h"][0],GAME_SETTINGS["enemy_speed_h"][1]) * self.enemy_speed_multiplier
        self.enemies[index][3] = random.uniform(GAME_SETTINGS["enemy_speed_v"][0],GAME_SETTINGS["enemy_speed_v"][1]) * self.enemy_speed_multiplier
        self.enemies[index][4] = random.randint(GAME_SETTINGS["enemy_cooldown"][0],GAME_SETTINGS["enemy_cooldown"][1])
        self.enemies[index][5] = random.randint(GAME_SETTINGS["enemy_cooldown"][0],GAME_SETTINGS["enemy_cooldown"][1])
        self.enemies[index][6] = random.randint(GAME_SETTINGS["enemy_health"][0],GAME_SETTINGS["enemy_health"][1])
        self.enemies[index][7] = self.enemies[index][6]

    def spawn_enemy_bullet(self,x,y):
        if self.e_bullet_count >= GAME_SETTINGS["max_bullets"]:
            return

        index = np.where(self.e_bullets[:,2] <= 0)[0][0]
        self.e_bullet_count += 1

        angle = atan2((self.player_y + self.player_bbox_2[1]) - y, (self.player_x + self.player_bbox_2[0]) - x)
        
        self.e_bullets[index][0] = x
        self.e_bullets[index][1] = y
        self.e_bullets[index][2] = 1
        self.e_bullets[index][3] = degrees(angle)
        self.e_bullets[index][4] = GAME_SETTINGS["bullet_speed"] * cos (angle)
        self.e_bullets[index][5] = GAME_SETTINGS["bullet_speed"] * sin (angle)

    def update_bullets(self,dt,display):
        for i in range(GAME_SETTINGS["max_bullets"]):
            if self.e_bullets[i][2] > 0:

                self.e_bullets[i][0] += self.e_bullets[i][4] * dt / DEFAULT_DELTA
                self.e_bullets[i][1] += self.e_bullets[i][5] * dt / DEFAULT_DELTA

                if self.e_bullets[i][0] < self.bounds[0] or self.e_bullets[i][1] < self.bounds[1] or self.e_bullets[i][0] > self.bounds[2] or self.e_bullets[i][1] > self.bounds[3]:
                    self.e_bullet_count -= 1
                    self.e_bullets[i][2] = -1
                    continue

                rotated = pg.transform.rotate(self.bullet_img,270 - self.e_bullets[i][3])

                if self.player_shield > 0:
                    if self.collide_enemy_bullet_shield(self.e_bullets[i][0],self.e_bullets[i][1],rotated.get_width()/2,rotated.get_height()/2):
                        self.e_bullet_count -= 1
                        self.e_bullets[i][2] = -1

                        if self.shield_regen_delay > 0:
                            continue

                        self.player_shield -= GAME_SETTINGS["bullet_damage"]
                        if self.player_shield < 0:
                            self.player_health += self.player_shield
                            self.player_shield = 0
                            self.shield_regen_delay = GAME_SETTINGS["shield_regen_delay"]
                        continue

                if self.collide_enemy_bullet_player(self.e_bullets[i][0],self.e_bullets[i][1],rotated.get_width()/2,rotated.get_height()/2):
                    self.e_bullet_count -= 1
                    self.e_bullets[i][2] = -1
                    self.player_health -= GAME_SETTINGS["bullet_damage"]
                    continue

                display.blit(rotated,(self.e_bullets[i][0],self.e_bullets[i][1]))

            if self.p_bullets[i][2] > 0:

                self.p_bullets[i][0] += self.p_bullets[i][4] * dt / DEFAULT_DELTA
                self.p_bullets[i][1] += self.p_bullets[i][5] * dt / DEFAULT_DELTA

                if self.p_bullets[i][0] < self.bounds[0] or self.p_bullets[i][1] < self.bounds[1] or self.p_bullets[i][0] > self.bounds[2] or self.p_bullets[i][1] > self.bounds[3]:
                    self.p_bullet_count -= 1
                    self.p_bullets[i][2] = -1
                    continue

                rotated = pg.transform.rotate(self.bullet_img,270 - self.p_bullets[i][3])

                if self.collide_player_bullet_enemy(self.p_bullets[i][0],self.p_bullets[i][1],rotated.get_width()/2,rotated.get_height()/2):
                    self.p_bullet_count -= 1
                    self.p_bullets[i][2] = -1
                    continue

                display.blit(rotated,(self.p_bullets[i][0],self.p_bullets[i][1]))     

    def update_enemies(self,dt,display):
        for i in range(GAME_SETTINGS["max_enemies"]):
            if self.enemies[i][6] > 0:
                self.enemies[i][4] -= dt
                if self.enemies[i][4] < 0:
                    self.spawn_enemy_bullet(self.enemies[i][0] + self.enemy_bbox_2[0] ,self.enemies[i][1] + self.enemy_bbox_2[1])
                    self.enemies[i][4] = self.enemies[i][5]

                self.enemies[i][1] += self.enemies[i][3] * dt / DEFAULT_DELTA

                if self.enemies[i][1] > self.bounds[3] or np.isclose(self.enemies[i][1],self.bounds[3]):
                    self.enemies[i][6] = -1
                    self.current_enemy_count -=1
                    self.player_health -= GAME_SETTINGS["enemy_damage"]
                    continue

                self.enemies[i][0] += self.enemies[i][2] * dt / DEFAULT_DELTA

                if self.collide_enemy_player(self.enemies[i][0], self.enemies[i][1]):
                    self.enemies[i][6] = -1
                    self.current_enemy_count -=1
                    self.player_health -= GAME_SETTINGS["enemy_damage"]
                    continue                    

                next_x = self.enemies[i][0] + self.enemies[i][2]  * dt / DEFAULT_DELTA

                if next_x < self.bounds[0] or np.isclose(next_x,self.bounds[0]):
                    self.enemies[i][2] = abs(self.enemies[i][2])

                if next_x > self.bounds[2] - self.enemy_bbox[0] or np.isclose(next_x,self.bounds[2] - self.enemy_bbox[0]):
                    self.enemies[i][2] = -abs(self.enemies[i][2])


                display.blit(self.enemy_img,(self.enemies[i][0],self.enemies[i][1]))

                pg.draw.rect(display, (255,0,0), (self.enemies[i][0], self.enemies[i][1]+self.enemy_bbox[1]+10, self.enemy_bbox[1], 10),2)
                pg.draw.rect(display, (255,0,0), (self.enemies[i][0], self.enemies[i][1]+self.enemy_bbox[1]+10, self.enemies[i][6] / self.enemies[i][7] * self.enemy_bbox[1], 10))

    def player_mov_left(self):
        self.mov_left = not self.mov_left

    def player_mov_right(self):
        self.mov_right = not self.mov_right

    def player_attack(self):
        if self.player_do_attack:
            self.player_attack_delay = 0
            self.player_do_attack = False
            return
        self.player_do_attack = True

    def spawn_player_bullet(self,x,y):
        if self.p_bullet_count >= GAME_SETTINGS["max_bullets"]:
            return

        index = np.where(self.p_bullets[:,2] <= 0)[0][0]
        self.p_bullet_count += 1
        
        self.p_bullets[index][0] = x - self.bullet_bbox_2[0]
        self.p_bullets[index][1] = y - self.bullet_bbox_2[1]
        self.p_bullets[index][2] = 1
        self.p_bullets[index][3] = -90
        self.p_bullets[index][4] = 0
        self.p_bullets[index][5] = - GAME_SETTINGS["bullet_speed"] * GAME_SETTINGS["player_bullet_speed_mul"]

    def update_player_params(self,dt):
        if self.mov_left or self.mov_right:
            if self.mov_right:
                if self.player_xvel < GAME_SETTINGS["player_top_speed"]:
                    self.player_xvel += GAME_SETTINGS["player_acceleration"] * dt / DEFAULT_DELTA
            if self.mov_left:
                if self.player_xvel > -GAME_SETTINGS["player_top_speed"]:
                    self.player_xvel -= GAME_SETTINGS["player_acceleration"] * dt / DEFAULT_DELTA
        else:
            value = self.player_xvel - copysign(1,self.player_xvel) * GAME_SETTINGS["player_deceleration"] * dt / DEFAULT_DELTA
            if copysign(1,value) == copysign(1,self.player_xvel):
                self.player_xvel = value
            else:
                self.player_xvel = 0

        next_x = self.player_x + self.player_xvel
        do_mov = True
        if next_x < self.bounds[0] or np.isclose(next_x,self.bounds[0]):
            self.player_xvel = 0
            do_mov = False

        if next_x > self.bounds[2] - self.enemy_bbox[0] or np.isclose(next_x,self.bounds[2] - self.enemy_bbox[0]):
            self.player_xvel = 0
            do_mov = False

        if do_mov:
            self.player_x = next_x

        if self.player_do_attack:
            self.player_attack_delay -= dt
            if self.player_attack_delay < 0:
                self.player_attack_delay = GAME_SETTINGS["player_cooldown"]
                self.spawn_player_bullet(self.player_x + self.player_bbox_2[0] ,self.player_y + self.player_bbox_2[1])


    def update_player_graphics(self,dt,display):

        if self.shield_charge_seq:
            if self.shield_regen_delay < 0:
                if self.player_shield < GAME_SETTINGS["player_shield"]:
                    self.player_shield += GAME_SETTINGS["shield_regen"]
            else:
                self.shield_regen_delay -= dt
        else:
            if self.player_shield < GAME_SETTINGS["player_shield"]:
                self.player_shield += GAME_SETTINGS["shield_regen"] * 2.6 * dt / DEFAULT_DELTA
            else:
                self.shield_charge_seq = True

        #self.player_health = 1000

        display.blit(self.player_img,(self.player_x,self.player_y))

    def display_player_hud(self,display):
        pg.draw.rect(display, (255,0,0), (self.hud_x, self.hud_health_y, self.hud_width * self.player_health / GAME_SETTINGS["player_health"], self.hud_height))
        pg.draw.rect(display, (255,0,0), (self.hud_x, self.hud_health_y, self.hud_width, self.hud_height),2)

        pg.draw.rect(display, (0,0,255), (self.hud_x, self.hud_shield_y, self.hud_width * self.player_shield / GAME_SETTINGS["player_shield"], self.hud_height))
        pg.draw.rect(display, (0,0,255), (self.hud_x, self.hud_shield_y, self.hud_width, self.hud_height),2)
                

    def run(self,dt,display,debug = False, font = None):

        self.enemy_speed_multiplier += dt * self.enemy_speed_multiplier_coeff

        if self.current_enemy_count < GAME_SETTINGS["max_enemies"]:
            self.current_spawn_delay -= dt
            if self.current_spawn_delay < 0:
                self.current_spawn_delay = random.randint(GAME_SETTINGS["enemy_delay"][0],GAME_SETTINGS["enemy_delay"][1])
                self.spawn_enemy()

        self.update_player_params(dt)
        self.update_bullets(dt,display)
        self.update_enemies(dt,display)
        self.shield.update_shield(display, self.player_x + self.player_bbox_2[0], self.player_y + self.player_bbox_2[1])
        self.update_player_graphics(dt,display)

        self.display_player_hud(display)

        if debug:
            pg.draw.rect(display, (255,255,0), (self.player_x-5 + self.player_bbox_2[0], self.player_y-5 + self.player_bbox_2[1],10,10))
            for i in range(self.shield.emiter_count):
                pg.draw.rect(display, (i*10+30,0,0), (self.player_x + self.player_bbox_2[0] + self.shield.emiter_centers[i][0] - self.shield.emiter_spacing/2 - self.shield.shield_radius, self.player_y + self.player_bbox_2[1] + self.shield.emiter_centers[i][1] - self.shield.emiter_spacing/2 - self.shield.shield_radius, self.shield.emiter_spacing, self.shield.emiter_spacing),1)
                fps_img = font.render(f'{self.shield.emiter_energy[i]:.4f}',True,(255,255,255))
                display.blit(fps_img,(self.player_x + self.player_bbox_2[0] + self.shield.emiter_centers[i][0] -  self.shield.shield_radius - 6, self.player_y + self.player_bbox_2[1] + self.shield.emiter_centers[i][1] - self.shield.shield_radius - 6))
        
        if self.player_health <= 0:
            self.reset()

class Application:
    def __init__(self):
        random.seed(time.time)
        pg.init()
        pg.font.init()
        self.display = pg.display.set_mode(APP_CONFIG["screen"])

        if (exists(GAME_RESOURCES["img"]["icon"])):
            self.load_by_relative = True
            self.dir = ''
        else:
            self.dir = dirname(__file__)
            self.load_by_relative = False

        self.show_info = False

        pg.display.set_caption(APP_CONFIG["title"])


        self.game = GameEngine([0,0]+APP_CONFIG["screen"])
        self.font = pg.font.SysFont(None, 14)
        self.clock = pg.time.Clock()
        self.target_fps = APP_CONFIG["fps"]

        if self.load_by_relative:
            pg.display.set_icon(pg.image.load(GAME_RESOURCES["img"]["icon"]))
        else:
            pg.display.set_icon(pg.image.load(f'{self.dir}/{GAME_RESOURCES["img"]["icon"]}'))

        self.game.load_resources(self.load_by_relative,self.dir)

    def run(self):
        running = True

        while running:
            dt = self.clock.tick(self.target_fps)
            self.display.fill((0,0,0))
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_F1:
                        self.show_info = not self.show_info

                    if event.key == pg.K_F2:
                        self.game.shield.change_mode()

                    if event.key == pg.K_a or event.key == pg.K_LEFT:
                        self.game.player_mov_left()
                    if event.key == pg.K_d or event.key == pg.K_RIGHT:
                        self.game.player_mov_right()
                    if event.key == pg.K_w or event.key == pg.K_UP or event.key == pg.K_SPACE:
                        self.game.player_attack()

                if event.type == pg.KEYUP:
                    if event.key == pg.K_a or event.key == pg.K_LEFT:
                        self.game.player_mov_left()
                    if event.key == pg.K_d or event.key == pg.K_RIGHT:
                        self.game.player_mov_right()
                    if event.key == pg.K_w or event.key == pg.K_UP or event.key == pg.K_SPACE:
                        self.game.player_attack()
            self.game.run(dt,self.display,self.show_info,self.font)

            if self.show_info:
                fps_img = self.font.render(f'FPS {int(self.clock.get_fps())}',True,(255,255,255))
                self.display.blit(fps_img,(0,0))
                fps_img = self.font.render(f'TDL {self.clock.get_time()}',True,(255,255,255))
                self.display.blit(fps_img,(0,14))
                fps_img = self.font.render(f'EDL {self.game.current_spawn_delay}',True,(255,255,255))
                self.display.blit(fps_img,(0,28))
                fps_img = self.font.render(f'ES% {self.game.enemy_speed_multiplier * 100:.2f}',True,(255,255,255))
                self.display.blit(fps_img,(0,42))

            pg.display.flip()

        pg.quit()

if __name__ == "__main__":
    app = Application()
    app.run()