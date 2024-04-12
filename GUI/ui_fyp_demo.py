import customtkinter
import os
from PIL import Image

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("GUI - Drone Classification System")
        self.geometry("1280x720")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images with light and dark mode image
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ui_images")
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "camera_drone.png")), size=(26, 26))
        self.home_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "home_dark.png")),
                                                 dark_image=Image.open(os.path.join(image_path, "home_light.png")), size=(25, 25))
        self.drone_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "drone_dark.png")),
                                                 dark_image=Image.open(os.path.join(image_path, "drone_light.png")), size=(25, 25))
        self.about_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "group_dark.png")),
                                                     dark_image=Image.open(os.path.join(image_path, "group_light.png")), size=(25, 25))
        self.process_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "process_dark.png")),
                                                     dark_image=Image.open(os.path.join(image_path, "process_light.png")), size=(20, 20))
        
        self.drone_1_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "drone1.jpg")), size=(200, 200))
        self.drone_2_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "drone2.jpg")), size=(200, 200))
        self.drone_3_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "drone3.jpg")), size=(200, 200))
        self.drone_4_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "drone4.png")), size=(200, 200))

        self.supervisor_1_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "supervisor1.jpeg")), size=(100, 100))
        self.supervisor_2_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "supervisor2.png")), size=(100, 100))
        self.supervisor_3_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "supervisor3.jpg")), size=(100, 100))

        self.member_1_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "member1.jpg")), size=(100, 100))
        self.member_2_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "member2.jpeg")), size=(100, 100))
        self.member_3_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "member3.jpeg")), size=(100, 100))
        self.member_4_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "member4.jpeg")), size=(100, 100))

        # navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(8, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="  Final Year Project", image=self.logo_image,
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=60, border_spacing=10, text="Home",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                   font=customtkinter.CTkFont(size=20), image=self.home_image, anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.info_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=60, border_spacing=10, text="Info",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                   font=customtkinter.CTkFont(size=20), image=self.drone_image, anchor="w", command=self.info_button_event)
        self.info_button.grid(row=2, column=0, sticky="ew")

        self.about_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=60, border_spacing=10, text="About Us",
                                                    fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                    font=customtkinter.CTkFont(size=20), image=self.about_image, anchor="w", command=self.about_button_event)
        self.about_button.grid(row=3, column=0, sticky="ew")

        ## navigation pane - appearance
        self.appearance_label = customtkinter.CTkLabel(self.navigation_frame, text="Appearance Mode:", anchor="w")
        self.appearance_label.grid(row=9, column=0, padx=20, pady=0)
        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=10, column=0, padx=20, pady=(0,20), sticky="s")

        ## navigation pane - scaling
        self.scaling_label = customtkinter.CTkLabel(self.navigation_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=11, column=0, padx=20, pady=0)
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=12, column=0, padx=20, pady=(0,20))

        # home frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure((0,1,2), weight=1)

        self.home_frame_large_label = customtkinter.CTkLabel(self.home_frame, text="Drone Detection & Classification System",
                                                             text_color=("gray10", "gray90"), font=customtkinter.CTkFont(size=40, weight="bold"))
        self.home_frame_large_label.grid(row=0, column=0, padx=0, pady=10, columnspan=3)

        ## home frame - configuration
        self.config_1_label = customtkinter.CTkLabel(self.home_frame, text="Samples per chirp:", font=customtkinter.CTkFont(size=15), text_color=("gray10", "gray90"), anchor="e")
        self.config_1_label.grid(row=1, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")
        self.config_1_entry = customtkinter.CTkEntry(self.home_frame)
        self.config_1_entry.insert(0, "256")
        self.config_1_entry.grid(row=1, column=1, padx=(20, 40), pady=(10, 10), sticky="w")

        self.config_2_label = customtkinter.CTkLabel(self.home_frame, text="Chirps per frame:", font=customtkinter.CTkFont(size=15), text_color=("gray10", "gray90"), anchor="e")
        self.config_2_label.grid(row=2, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")
        self.config_2_entry = customtkinter.CTkEntry(self.home_frame)
        self.config_2_entry.insert(0, "128")
        self.config_2_entry.grid(row=2, column=1, padx=(20, 40), pady=(10, 10), sticky="w")

        self.config_3_label = customtkinter.CTkLabel(self.home_frame, text="Number of frames:", font=customtkinter.CTkFont(size=15), text_color=("gray10", "gray90"), anchor="e")
        self.config_3_label.grid(row=3, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")
        self.config_3_entry = customtkinter.CTkEntry(self.home_frame)
        self.config_3_entry.insert(0, "256")
        self.config_3_entry.grid(row=3, column=1, padx=(20, 40), pady=(10, 10), sticky="w")

        self.config_4_label = customtkinter.CTkLabel(self.home_frame, text="Frame rate:", font=customtkinter.CTkFont(size=15), text_color=("gray10", "gray90"), anchor="e")
        self.config_4_label.grid(row=4, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")
        self.config_4_entry = customtkinter.CTkEntry(self.home_frame)
        self.config_4_entry.insert(0, "25Hz")
        self.config_4_entry.grid(row=4, column=1, padx=(20, 40), pady=(10, 10), sticky="w")

        ## home frame - file paths entry
        self.audio_entry_label = customtkinter.CTkLabel(self.home_frame, text="Audio File Path:", font=customtkinter.CTkFont(size=15), text_color=("gray10", "gray90"), anchor="e")
        self.audio_entry_label.grid(row=5, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")
        self.audio_entry = customtkinter.CTkEntry(self.home_frame, placeholder_text="Audio File Path")
        self.audio_entry.grid(row=5, column=1, padx=(20, 40), pady=(10, 10), sticky="nsew")

        self.radar_entry_label = customtkinter.CTkLabel(self.home_frame, text="Radar Data File Path:", font=customtkinter.CTkFont(size=15), text_color=("gray10", "gray90"), anchor="e")
        self.radar_entry_label.grid(row=6, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")
        self.radar_entry = customtkinter.CTkEntry(self.home_frame, placeholder_text="Radar Data File Path")
        self.radar_entry.grid(row=6, column=1, padx=(20, 40), pady=(10, 10), sticky="nsew")

        ## home frame - process button
        self.home_process_button = customtkinter.CTkButton(self.home_frame, text="Process", text_color=("gray10", "gray90"),
                                                           font=customtkinter.CTkFont(size=20), image=self.process_image, compound="right", command=self.process_button_event)
        self.home_process_button.grid(row=7, column=0, padx=20, pady=20, columnspan=3)

        ## home frame - results
        self.drone_exist_label = customtkinter.CTkLabel(self.home_frame, text="Does a drone exist:", font=customtkinter.CTkFont(size=15), text_color=("gray10", "gray90"), anchor="e")
        self.drone_exist_label.grid(row=8, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")
        self.drone_exist_val_frame = customtkinter.CTkFrame(master=self.home_frame, border_width=1, border_color=("gray10", "gray90"))
        self.drone_exist_val_frame.grid(row=8, column=1, padx=(20, 20), pady=(10, 10), sticky="nsew")
        self.drone_exist_val_label = customtkinter.CTkLabel(self.drone_exist_val_frame, text="Nan", font=customtkinter.CTkFont(size=15), text_color=("gray10", "gray90"), anchor="w")
        self.drone_exist_val_label.pack(padx=10, pady=10)

        self.drone_type_label = customtkinter.CTkLabel(self.home_frame, text="Drone type:", font=customtkinter.CTkFont(size=15), text_color=("gray10", "gray90"), anchor="e")
        self.drone_type_label.grid(row=9, column=0, padx=(20, 20), pady=(10, 10), sticky="nsew")
        self.drone_type_val_frame = customtkinter.CTkFrame(master=self.home_frame, border_width=1, border_color=("gray10", "gray90"))
        self.drone_type_val_frame.grid(row=9, column=1, padx=(20, 20), pady=(10, 10), sticky="nsew")
        self.drone_type_val_label = customtkinter.CTkLabel(self.drone_type_val_frame, text="Nan", font=customtkinter.CTkFont(size=15), text_color=("gray10", "gray90"), anchor="w")
        self.drone_type_val_label.pack(padx=10, pady=10)

        # info frame
        self.info_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.info_frame.grid_columnconfigure((0,1), weight=1)

        self.info_title_label = customtkinter.CTkLabel(self.info_frame, text="Drone Types", font=customtkinter.CTkFont(size=25), text_color=("gray10", "gray90"))
        self.info_title_label.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), columnspan=2)

        self.drone_1_label = customtkinter.CTkLabel(self.info_frame, text="DJI Matrice 300 RTK", image=self.drone_1_image, compound="top", font=customtkinter.CTkFont(size=15))
        self.drone_1_label.grid(row=1, column=0, padx=20, pady=20)
        self.drone_2_label = customtkinter.CTkLabel(self.info_frame, text="DJI Phanthom 4 Pro Plus", image=self.drone_2_image, compound="top", font=customtkinter.CTkFont(size=15))
        self.drone_2_label.grid(row=1, column=1, padx=20, pady=20)
        self.drone_3_label = customtkinter.CTkLabel(self.info_frame, text="Mavic 2 Enterprise Dual", image=self.drone_3_image, compound="top", font=customtkinter.CTkFont(size=15))
        self.drone_3_label.grid(row=2, column=0, padx=20, pady=20)
        self.drone_4_label = customtkinter.CTkLabel(self.info_frame, text="Custom Drone", image=self.drone_4_image, compound="top", font=customtkinter.CTkFont(size=15))
        self.drone_4_label.grid(row=2, column=1, padx=20, pady=20)

        # about frame
        self.about_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.about_frame.grid_columnconfigure((0,1,2,3), weight=1)

        self.about_title_label = customtkinter.CTkLabel(self.about_frame, text="Our Team", font=customtkinter.CTkFont(size=25), text_color=("gray10", "gray90"))
        self.about_title_label.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), columnspan=4)

        self.supervisor_1_label = customtkinter.CTkLabel(self.about_frame, text="Dr. Sampath Perera\n(Supervisor)", image=self.supervisor_1_image, compound="top", font=customtkinter.CTkFont(size=15))
        self.supervisor_1_label.grid(row=1, column=0, padx=0, pady=20, columnspan=4)
        self.supervisor_2_label = customtkinter.CTkLabel(self.about_frame, text="Dr. Ranga Rodrigo\n(Co-supervisor)", image=self.supervisor_2_image, compound="top", font=customtkinter.CTkFont(size=15))
        self.supervisor_2_label.grid(row=2, column=0, padx=20, pady=20, columnspan=2)
        self.supervisor_3_label = customtkinter.CTkLabel(self.about_frame, text="Dr. Chamira U. S. Edussooriya\n(Co-supervisor)", image=self.supervisor_3_image, compound="top", font=customtkinter.CTkFont(size=15))
        self.supervisor_3_label.grid(row=2, column=2, padx=20, pady=20, columnspan=2)

        self.member_1_label = customtkinter.CTkLabel(self.about_frame, text="T.P. Sankalpa\n(Team Leader, Member)", image=self.member_1_image, compound="top", font=customtkinter.CTkFont(size=15))
        self.member_1_label.grid(row=3, column=0, padx=20, pady=20)
        self.member_2_label = customtkinter.CTkLabel(self.about_frame, text="W.K.G.G. Sumanasekara\n(Member)", image=self.member_2_image, compound="top", font=customtkinter.CTkFont(size=15))
        self.member_2_label.grid(row=3, column=1, padx=20, pady=20)
        self.member_3_label = customtkinter.CTkLabel(self.about_frame, text="K.K.S. Punsara\n(Member)", image=self.member_3_image, compound="top", font=customtkinter.CTkFont(size=15))
        self.member_3_label.grid(row=3, column=2, padx=20, pady=20)
        self.member_4_label = customtkinter.CTkLabel(self.about_frame, text="D.P.G. Hettihewa\n(Member)", image=self.member_4_image, compound="top", font=customtkinter.CTkFont(size=15))
        self.member_4_label.grid(row=3, column=3, padx=20, pady=20)

        # select default values
        self.select_frame_by_name("home")
        self.change_scaling_event("100%")
        self.scaling_optionemenu.set("100%")
        self.change_appearance_mode_event("System") # Modes: "System" (standard), "Dark", "Light"
        self.appearance_mode_menu.set("System")
        customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.info_button.configure(fg_color=("gray75", "gray25") if name == "info" else "transparent")
        self.about_button.configure(fg_color=("gray75", "gray25") if name == "about" else "transparent")

        # show selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "info":
            self.info_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.info_frame.grid_forget()
        if name == "about":
            self.about_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.about_frame.grid_forget()

    def home_button_event(self):
        self.select_frame_by_name("home")

    def info_button_event(self):
        self.select_frame_by_name("info")

    def about_button_event(self):
        self.select_frame_by_name("about")

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)
    
    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
    
    def process_button_event(self):
        print("process button clicked")
        config_1, config_2, config_3, config_4 = self.config_1_entry.get(), self.config_2_entry.get(), self.config_3_entry.get(), self.config_4_entry.get()
        path_1, path_2 = self.audio_entry.get(), self.radar_entry.get()
        self.drone_exist_val_label.configure(text="Yes")
        self.drone_type_val_label.configure(text="DJ Mavic Mini")

        print(config_1, config_2, config_3, config_4)
        print(path_1, path_2)


if __name__ == "__main__":
    app = App()
    app.mainloop()