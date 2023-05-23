from numpy.random import seed
from numpy.random import randint
import numpy as np
import random
import time
import threading
import pygame

#seed(1)
class Junction(threading.Thread):
    def  __init__(self, tc_max, obj):
        threading.Thread.__init__(self)
        self.tc_max = tc_max
        self.counts = randint(10, 25, 4).tolist()
        self.effectiveness = [0, 0, 0, 0]
        self.lane_access_count = [0, 0, 0, 0]
        self.access_lane = None
        self.min_access_time = 2
        self.max_access_time = 20
        self.adjacent_junctions = None
        self.obj = obj
        self.tc_ratio_difference = None
        self.incoming = None
    def set_adjacent_junctions(self, adjacent_junctions):
         self.adjacent_junctions = adjacent_junctions
        
    def Tc_ratio(self):
        Tc = sum(self.counts)
        return round(Tc/self.tc_max,2)

    def get_tc_ratio_differences(self):
        arr = []
        #print("*******"+str(self.Tc_ratio())+"***********")
        for junction in self.adjacent_junctions:
            if(junction is None):
                arr.append(self.Tc_ratio() - round(random.uniform(0, 0.4), 4))
            else:
                arr.append((self.Tc_ratio() - junction.Tc_ratio()))
                #print("******"+str(self.Tc_ratio() - junction.Tc_ratio())+"*********")
        return arr

    def polynomial(self, tc_ratio_difference, effectivity, lane_access_count):
        a, b, c = 0.7, 0.9, -0.9
        #print("******"+str(a*effectivity )+"*********")
        return a*effectivity + b*tc_ratio_difference + c * (lane_access_count)**0.5
        
    def select_lane(self):
        self.tc_ratio_difference = self.get_tc_ratio_differences()
        max_val = 0
        max_val_index = None
        lane_access_count_relative = [None, None, None, None]
        min_access_count = min(self.lane_access_count)
        if min_access_count > 3:
            min_access_count -= 3
        
        for i in range(4):
            lane_access_count_relative[i] = self.lane_access_count[i] - min_access_count
            
        for i in range(4):
            val = self.polynomial(self.tc_ratio_difference[i], self.effectiveness[i], lane_access_count_relative[i])*10
            #print(" "+self.obj+" "+str(val))
            if(val > max_val):
                max_val = val
                max_val_index = i
        return max_val, max_val_index

    def run(self):
        while(True):
            access_time, self.access_lane = self.select_lane()
            ls1 = []
            for i in range(4):
                if(self.adjacent_junctions[i] is None and self.tc_ratio_difference[i]>0):
                    #print("*******"+str(self.tc_ratio_difference[i])+"***********")
                    ls1.append(i)
                    
            if(len(ls1)!=0):
                r_choice = random.choice(ls1)
                self.incoming = r_choice
                self.counts[r_choice] += random.randint(5, 10)
            
            
            if(self.access_lane == None):
                #print(" "+self.obj+" Halt "+str(self.Tc_ratio())+" ")
                continue
            
            symb = ""
            if(self.access_lane == 0):
                symb = "v"
            elif(self.access_lane == 1):
                symb = "<"
            elif(self.access_lane == 2):
                symb = "^"
            else:
                symb = ">"
            
            all_rem = 0
            
            #print(self.obj+" access time "+str(access_time)+" ")
            time.sleep(1)
            for i in range(4):
                if(i == self.access_lane):
                    continue
                rem = int(self.counts[i]*access_time/115)
                #print("*****"+str(rem)+"******")
                #print(print(self.obj+" rem "+str(rem)+" "))
                if(self.counts[i] > rem):
                    self.counts[i] -= rem
                    all_rem += rem
                else:
                    self.counts[i] -= int(self.counts[i]/3)
                    all_rem += int(self.counts[i]/3)
            #print(self.obj+" all rem "+str(all_rem)+" ")
            
            #print("*****"+str(all_rem)+"*****")
            if(self.adjacent_junctions[self.access_lane] is not None):        
                self.adjacent_junctions[self.access_lane].counts[(self.access_lane+2)%4] += all_rem
                #print(self.adjacent_junctions[self.access_lane].obj)
                self.adjacent_junctions[self.access_lane].incoming = (self.access_lane+2)%4
            #print(self.obj+" "+str(all_rem)+" "+str(self.access_lane)+" "+" ".join(map(str,self.lane_access_count))+" "+" ".join(map(str,self.counts))+" "+" ".join(map(str,self.effectiveness))+" ")
            self.lane_access_count[self.access_lane] += 1
            self.effectiveness[self.access_lane] = 0.6*(all_rem)
            #self.effectiveness
            #print(self.obj+" "+str(self.Tc_ratio())+" "+symb+" ")
            
class Simulation(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.centre = Junction(20, 'centre')
        self.down = Junction(30, 'down')
        self.left = Junction(30, 'left')
        self.up = Junction(30, 'up')
        self.right = Junction(30, 'right')
        self.down.set_adjacent_junctions([None, None, self.centre, None])
        self.left.set_adjacent_junctions([None, None, None, self.centre])
        self.up.set_adjacent_junctions([self.centre, None, None, None])
        self.right.set_adjacent_junctions([None, self.centre, None, None])
        self.centre.set_adjacent_junctions([self.down, self.left, self.up, self.right])
        self.centre.start()
        self.down.start()
        self.left.start()
        self.up.start()
        self.right.start()
        self.X = int(3070/2)
        self.Y = int(2249/2)
        pygame.init()
        self.scrn = pygame.display.set_mode((self.X, self.Y))
        pygame.display.set_caption('simulation')
        self.imp = pygame.image.load("D:/Simulation/Junctions1.png").convert()
        self.imp = pygame.transform.scale(self.imp, (self.X, self.Y))
        self.scrn.blit(self.imp, (0, 0))
        pygame.display.flip()
        self.status = True
        self.white = (255,255,255)
        self.red = (255,0,0)
        self.green = (0,255,0)
        self.blue = (0,0,255)
        self.black = (0,0,0)
        
    def draw_arrow(self,
        surface: pygame.Surface,
        start: pygame.Vector2,
        end: pygame.Vector2,
        color: pygame.Color,
        body_width: int = 20,
        head_width: int = 40,
        head_height: int = 20
    ):
        arrow = start - end
        angle = arrow.angle_to(pygame.Vector2(0, -1))
        body_length = arrow.length() - head_height

        # Create the triangle head around the origin
        head_verts = [
            pygame.Vector2(0, head_height / 2),  # Center
            pygame.Vector2(head_width / 2, -head_height / 2),  # Bottomright
            pygame.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
        ]
        # Rotate and translate the head into place
        translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(-angle)
        for i in range(len(head_verts)):
            head_verts[i].rotate_ip(-angle)
            head_verts[i] += translation
            head_verts[i] += start

        pygame.draw.polygon(surface, color, head_verts)

        # Stop weird shapes when the arrow is shorter than arrow head
        if arrow.length() >= head_height:
            # Calculate the body rect, rotate and translate into place
            body_verts = [
                pygame.Vector2(-body_width / 2, body_length / 2),  # Topleft
                pygame.Vector2(body_width / 2, body_length / 2),  # Topright
                pygame.Vector2(body_width / 2, -body_length / 2),  # Bottomright
                pygame.Vector2(-body_width / 2, -body_length / 2),  # Bottomleft
            ]
            translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
            for i in range(len(body_verts)):
                body_verts[i].rotate_ip(-angle)
                body_verts[i] += translation
                body_verts[i] += start

            pygame.draw.polygon(surface, color, body_verts)
        
    def run(self):
        font = pygame.font.SysFont("Arial", 36)
        font2 = pygame.font.SysFont("Times New Roman", 36)
        font1 = pygame.font.SysFont("Arial", 40)
        while(True):
            self.scrn.blit(self.imp, (0, 0))

        ##left
            left_counts = self.left.counts
            left_effectiveness = self.left.lane_access_count
            tc_ratio_left = self.left.Tc_ratio()
            tc_ratio_left1 = font.render(str(tc_ratio_left), True, self.black)
            left_d = font.render(str(left_counts[0]), True, self.red)
            left_l = font.render(str(left_counts[1]), True, self.red)
            left_u = font.render(str(left_counts[2]), True, self.red)
            left_r = font.render(str(left_counts[3]), True, self.red)
            left_d_e = font2.render("["+str(left_effectiveness[0])+"]", True, self.blue)
            left_l_e = font2.render("["+str(left_effectiveness[1])+"]", True, self.blue)
            left_u_e = font2.render("["+str(left_effectiveness[2])+"]", True, self.blue)
            left_r_e = font2.render("["+str(left_effectiveness[3])+"]", True, self.blue)

            self.scrn.blit(left_d,((399, 678)))
            self.scrn.blit(left_l,((321, 548)))
            self.scrn.blit(left_u,((455, 475)))
            self.scrn.blit(left_r,((536, 603)))
            
            self.scrn.blit(left_d_e,((458, 678)))
            self.scrn.blit(left_l_e,((315, 603)))
            self.scrn.blit(left_u_e,((395, 475)))
            self.scrn.blit(left_r_e,((536, 548)))
            end_l = None
            center_l = pygame.Vector2(445, 572)
            if(self.left.access_lane is None):
                txtsurf = font1.render("HALT", True, self.blue)
                self.scrn.blit(txtsurf,((418, 558)))
            else:
                if(self.left.access_lane == 0):
                    end_l = pygame.Vector2(445, 612)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1471, 1001), pygame.Vector2(1471, 1050), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
                elif(self.left.access_lane == 1):
                    end_l = pygame.Vector2(405, 572)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1108, 767), pygame.Vector2(1060, 767), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
                elif(self.left.access_lane == 2):
                    end_l = pygame.Vector2(445, 532)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1347, 462), pygame.Vector2(1347, 410), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
                else:
                    end_l = pygame.Vector2(485, 572)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1699, 650), pygame.Vector2(1750, 650), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
            self.draw_arrow(self.scrn, center_l, end_l, pygame.Color("dodgerblue"), 20, 40, 30)
            self.scrn.blit(tc_ratio_left1,((421, 610)))
            
        ##right
            right_counts = self.right.counts
            right_effectiveness = self.right.lane_access_count
            tc_ratio_right = self.right.Tc_ratio()
            tc_ratio_right1 = font.render(str(tc_ratio_right), True, self.black)
            right_d = font.render(str(right_counts[0]), True, self.red)
            right_l = font.render(str(right_counts[1]), True, self.red)
            right_u = font.render(str(right_counts[2]), True, self.red)
            right_r = font.render(str(right_counts[3]), True, self.red)
            right_d_e = font2.render("["+str(right_effectiveness[0])+"]", True, self.blue)
            right_l_e = font2.render("["+str(right_effectiveness[1])+"]", True, self.blue)
            right_u_e = font2.render("["+str(right_effectiveness[2])+"]", True, self.blue)
            right_r_e = font2.render("["+str(right_effectiveness[3])+"]", True, self.blue)

            self.scrn.blit(right_d,((1044, 683)))
            self.scrn.blit(right_l,((964, 549)))
            self.scrn.blit(right_u,((1100, 475)))
            self.scrn.blit(right_r,((1181, 601)))
            
            self.scrn.blit(right_d_e,((1101, 683)))
            self.scrn.blit(right_l_e,((962, 604)))
            self.scrn.blit(right_u_e,((1038, 475)))
            self.scrn.blit(right_r_e,((1187, 546)))
            end_r = None
            
            center_r = pygame.Vector2(1089, 572)
            if(self.right.access_lane is None):
                txtsurf = font1.render("HALT", True, self.blue)
                self.scrn.blit(txtsurf,((1061, 565)))
            else:
                if(self.right.access_lane == 0):
                    end_r = pygame.Vector2(1089, 612)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1471, 1001), pygame.Vector2(1471, 1050), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
                elif(self.right.access_lane == 1):
                    end_r = pygame.Vector2(1049, 572)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1108, 767), pygame.Vector2(1060, 767), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
                elif(self.right.access_lane == 2):
                    end_r = pygame.Vector2(1089, 532)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1347, 462), pygame.Vector2(1347, 410), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
                else:
                    end_r = pygame.Vector2(1129, 572)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1699, 650), pygame.Vector2(1750, 650), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
            self.draw_arrow(self.scrn, center_r, end_r, pygame.Color("dodgerblue"), 20, 40, 30)
            self.scrn.blit(tc_ratio_right1,((1064, 610)))
            
        ##up
            up_counts = self.up.counts
            up_effectiveness = self.up.lane_access_count
            tc_ratio_up = self.up.Tc_ratio()
            tc_ratio_up1 = font.render(str(tc_ratio_up), True, self.black)
            up_d = font.render(str(up_counts[0]), True, self.red)
            up_l = font.render(str(up_counts[1]), True, self.red)
            up_u = font.render(str(up_counts[2]), True, self.red)
            up_r = font.render(str(up_counts[3]), True, self.red)
            up_d_e = font2.render("["+str(up_effectiveness[0])+"]", True, self.blue)
            up_l_e = font2.render("["+str(up_effectiveness[1])+"]", True, self.blue)
            up_u_e = font2.render("["+str(up_effectiveness[2])+"]", True, self.blue)
            up_r_e = font2.render("["+str(up_effectiveness[3])+"]", True, self.blue)

            self.scrn.blit(up_d,((720, 371)))
            self.scrn.blit(up_l,((644, 242)))
            self.scrn.blit(up_u,((780, 171)))
            self.scrn.blit(up_r,((860, 296)))
            
            self.scrn.blit(up_d_e,((783, 371)))
            self.scrn.blit(up_l_e,((646, 296)))
            self.scrn.blit(up_u_e,((719, 171)))
            self.scrn.blit(up_r_e,((858, 239)))
            
            end_u = None
            center_u = pygame.Vector2(766, 264)
            if(self.up.access_lane is None):
                txtsurf = font1.render("HALT", True, self.blue)
                self.scrn.blit(txtsurf,((739, 257)))
            else:
                if(self.up.access_lane == 0):
                    end_u = pygame.Vector2(766, 304)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1471, 1001), pygame.Vector2(1471, 1050), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
                elif(self.up.access_lane == 1):
                    end_u = pygame.Vector2(726, 264)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1108, 767), pygame.Vector2(1060, 767), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
                elif(self.up.access_lane == 2):
                    end_u = pygame.Vector2(766, 224)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1347, 462), pygame.Vector2(1347, 410), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
                else:
                    end_u = pygame.Vector2(806, 264)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1699, 650), pygame.Vector2(1750, 650), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
            self.draw_arrow(self.scrn, center_u, end_u, pygame.Color("dodgerblue"), 20, 40, 30)
            self.scrn.blit(tc_ratio_up1,((736, 306)))
            
        ##down
            down_counts = self.down.counts
            down_effectiveness = self.down.lane_access_count
            tc_ratio_down = self.down.Tc_ratio()
            tc_ratio_down1 = font.render(str(tc_ratio_down), True, self.black)
            down_d = font.render(str(down_counts[0]), True, self.red)
            down_l = font.render(str(down_counts[1]), True, self.red)
            down_u = font.render(str(down_counts[2]), True, self.red)
            down_r = font.render(str(down_counts[3]), True, self.red)
            down_d_e = font2.render("["+str(down_effectiveness[0])+"]", True, self.blue)
            down_l_e = font2.render("["+str(down_effectiveness[1])+"]", True, self.blue)
            down_u_e = font2.render("["+str(down_effectiveness[2])+"]", True, self.blue)
            down_r_e = font2.render("["+str(down_effectiveness[3])+"]", True, self.blue)

            self.scrn.blit(down_d,((718, 958)))
            self.scrn.blit(down_l,((641, 856)))
            self.scrn.blit(down_u,((775, 777)))
            self.scrn.blit(down_r,((858, 912)))
            
            self.scrn.blit(down_d_e,((778, 958)))
            self.scrn.blit(down_l_e,((641, 912)))
            self.scrn.blit(down_u_e,((719, 777)))
            self.scrn.blit(down_r_e,((860, 855)))
            end_d = None

            center_d = pygame.Vector2(766, 887)
            if(self.down.access_lane is None):
                txtsurf = font1.render("HALT", True, self.blue)
                self.scrn.blit(txtsurf,((744, 875)))
            else:
                if(self.down.access_lane == 0):
                    end_d = pygame.Vector2(766, 927)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1471, 1001), pygame.Vector2(1471, 1050), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
                elif(self.down.access_lane == 1):
                    end_d = pygame.Vector2(726, 887)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1108, 767), pygame.Vector2(1060, 767), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
                elif(self.down.access_lane == 2):
                    end_d = pygame.Vector2(766, 847)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1347, 462), pygame.Vector2(1347, 410), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
                else:
                    end_d = pygame.Vector2(806, 887)
                    #self.draw_arrow(self.scrn, pygame.Vector2(1699, 650), pygame.Vector2(1750, 650), pygame.Color(random.choice(["blue","green", "yellow"])), 20, 40, 30)
            self.draw_arrow(self.scrn, center_d, end_d, pygame.Color("dodgerblue"), 20, 40, 30)
            self.scrn.blit(tc_ratio_down1,((736, 923)))
            
        ##centre
            centre_counts = self.centre.counts
            centre_effectiveness = self.centre.lane_access_count
            tc_ratio_centre = self.centre.Tc_ratio()
            tc_ratio_centre1 = font.render(str(tc_ratio_centre), True, self.black)
            centre_d = font.render(str(centre_counts[0]), True, self.red)
            centre_l = font.render(str(centre_counts[1]), True, self.red)
            centre_u = font.render(str(centre_counts[2]), True, self.red)
            centre_r = font.render(str(centre_counts[3]), True, self.red)
            centre_d_e = font2.render("["+str(centre_effectiveness[0])+ "]", True, self.blue)
            centre_l_e = font2.render("["+str(centre_effectiveness[1])+ "]", True, self.blue)
            centre_u_e = font2.render("["+str(centre_effectiveness[2])+ "]", True, self.blue)
            centre_r_e = font2.render("["+str(centre_effectiveness[3])+ "]", True, self.blue)

            self.scrn.blit(centre_d,((719, 680)))
            self.scrn.blit(centre_l,((641, 548)))
            self.scrn.blit(centre_u,((779, 475)))
            self.scrn.blit(centre_r,((856, 602)))
            self.scrn.blit(centre_d_e,((781, 680)))
            self.scrn.blit(centre_l_e,((639, 600)))
            self.scrn.blit(centre_u_e,((720, 475)))
            self.scrn.blit(centre_r_e,((856, 548)))
            center = pygame.Vector2((766, 572))
            end = None
            #print(self.centre.incoming)

            if self.centre.incoming is not None:
                if(self.centre.incoming == 0):
                    self.draw_arrow(self.scrn, pygame.Vector2(739, 755), pygame.Vector2(739, 715), pygame.Color(random.choice(["red", "black"])), 20, 40, 30)
                elif(self.centre.incoming == 1):
                    self.draw_arrow(self.scrn, pygame.Vector2(600, 562), pygame.Vector2(640, 562), pygame.Color(random.choice(["red", "black"])), 20, 40, 30)
                elif(self.centre.incoming == 2):
                    self.draw_arrow(self.scrn, pygame.Vector2(796, 436), pygame.Vector2(796, 476), pygame.Color(random.choice(["red", "black"])), 20, 40, 30)
                else:
                    self.draw_arrow(self.scrn, pygame.Vector2(929, 617), pygame.Vector2(890, 617), pygame.Color(random.choice(["red", "black"])), 20, 40, 30)
            if(self.centre.access_lane is None):
                txtsurf = font1.render("HALT", True, self.blue)
                self.scrn.blit(txtsurf,((730, 560)))
            
            else:
                if(self.centre.access_lane == 0):
                    end = pygame.Vector2(766, 612)
                    self.draw_arrow(self.scrn, pygame.Vector2(796, 736), pygame.Vector2(796, 776), pygame.Color(random.choice(["green","white"])), 20, 40, 30)
                elif(self.centre.access_lane == 1):
                    end = pygame.Vector2(726, 572)
                    self.draw_arrow(self.scrn, pygame.Vector2(616, 620), pygame.Vector2(576, 620), pygame.Color(random.choice(["green","white"])), 20, 40, 30)
                elif(self.centre.access_lane == 2):
                    end = pygame.Vector2(766, 532)
                    self.draw_arrow(self.scrn, pygame.Vector2(738, 444), pygame.Vector2(738, 404), pygame.Color(random.choice(["green","white"])), 20, 40, 30)
                else:
                    end = pygame.Vector2(806, 572)
                    self.draw_arrow(self.scrn, pygame.Vector2(919, 562), pygame.Vector2(959, 562), pygame.Color(random.choice(["green","white"])), 20, 40, 30)

                self.draw_arrow(self.scrn, center, end, pygame.Color("dodgerblue"), 20, 40, 30)
               
            self.scrn.blit(tc_ratio_centre1,((736, 610)))
            

            pygame.display.update()
            
s = Simulation()
s.start()
s.join()
