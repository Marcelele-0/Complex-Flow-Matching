from manim import *
import numpy as np

config.background_color = WHITE


class EuclideanFlow(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            axis_config={"color": BLACK, "include_tip": True},
            x_length=6,
            y_length=6,
        )
        labels = axes.get_axis_labels(
            x_label=Tex(r"$\mathrm{Re}(z)$", color=BLACK),
            y_label=Tex(r"$\mathrm{Im}(z)$", color=BLACK),
        )

        # Symmetrical complex points forcing linear trajectory directly through the origin (0, 0)
        p_start = axes.c2p(1.8, 2.0)
        p_end = axes.c2p(-1.8, -2.0)

        dot_start = Dot(p_start, color=BLUE, radius=0.1)
        dot_end = Dot(p_end, color=BLUE, radius=0.1)
        bad_path = Arrow(p_start, p_end, color=RED, buff=0.1, stroke_width=6)

        label_z0 = MathTex("z_0", color=BLACK).next_to(dot_start, UR, buff=0.1)
        label_z1 = MathTex("z_1", color=BLACK).next_to(dot_end, DL, buff=0.1)

        self.add(axes, labels, bad_path, dot_start, dot_end, label_z0, label_z1)


class CylindricalFlow(ThreeDScene):
    def construct(self):
        # Oddalamy kamere (zoom=0.6), zeby caly walec z opisami byl idealnie widoczny w kadrze
        # phi=68 deg, theta=180 deg
        self.set_camera_orientation(phi=68 * DEGREES, theta=180 * DEGREES, zoom=0.6)

        # Promienie
        R = 2.5  # Promien walca
        R_curve = R + 0.03  # Delikatne wysuniecie nad powierzchnie
        R_axis = 3.1
        Z_base = -3.0

        # Katy kluczowe (dokladnie 180 st odstepu miedzy faza 0 a faza pi/-pi)
        alpha_0 = 70 * DEGREES  # Lewa strona: Phase = 0 oraz Os Amplitudy
        alpha_pi = 250 * DEGREES  # Prawa strona: Szew pi / -pi (na przednio-prawej sciance walca)

        # Powierzchnia walca
        cylinder = Surface(
            lambda u, v: np.array([R * np.cos(u), R * np.sin(u), v]),
            u_range=[alpha_0 - PI, alpha_0 + PI],
            v_range=[-3, 3],
            checkerboard_colors=[BLUE_E, BLUE_C],
            resolution=(24, 12),
            fill_opacity=0.3,
            stroke_color=GRAY_B,
            stroke_width=0.5,
        )

        # Szew walca (linia przerywana laczaca podstawe z gora wzdluz alpha_pi)
        seam_line = DashedLine(
            start=np.array([R_curve * np.cos(alpha_pi), R_curve * np.sin(alpha_pi), -3.0]),
            end=np.array([R_curve * np.cos(alpha_pi), R_curve * np.sin(alpha_pi), 3.0]),
            dashed_ratio=0.5,
            dash_length=0.15,
            color=GRAY_D,
            stroke_width=2,
        )

        # --- Os Amplitudy (pionowa, przecinajaca okrag fazy w punkcie Phase = 0) ---
        p_amp_origin = np.array([R_axis * np.cos(alpha_0), R_axis * np.sin(alpha_0), Z_base])
        p_amp_top = np.array([R_axis * np.cos(alpha_0), R_axis * np.sin(alpha_0), 3.5])

        amp_axis = Arrow(start=p_amp_origin, end=p_amp_top, buff=0, color=BLACK, stroke_width=4)
        label_amp = Tex("Amplitude", color=BLACK).move_to(p_amp_top + np.array([0, 0, 0.4]))

        # --- Os Fazy (pelny, ciagly okrag S^1 u podstawy walca na poziomie Z_base = -3.0) ---
        phase_circle = ParametricFunction(
            lambda u: np.array([R_axis * np.cos(u), R_axis * np.sin(u), Z_base]),
            t_range=[0, TAU],
            color=BLACK,
            stroke_width=4,
        )

        # Grot wskazujacy kierunek wzrostu fazy (z przodu walca)
        u_phase_tip = 180 * DEGREES
        p_phase_tip = np.array([R_axis * np.cos(u_phase_tip), R_axis * np.sin(u_phase_tip), Z_base])
        t_phase = np.array([-np.sin(u_phase_tip), np.cos(u_phase_tip), 0])
        t_phase_u = t_phase / np.linalg.norm(t_phase)

        phase_tip = Cone(
            direction=t_phase_u,
            base_radius=0.11,
            height=0.28,
            show_base=True,
            fill_color=BLACK,
            stroke_color=BLACK,
            stroke_width=0,
            fill_opacity=1.0,
        )
        phase_tip.shift(p_phase_tip)

        # Znacznik 0 (punkt przeciecia osi amplitudy i okregu fazy po lewej stronie)
        dir_0 = np.array([np.cos(alpha_0), np.sin(alpha_0), 0])
        tick_0 = Line(
            p_amp_origin - 0.16 * dir_0, p_amp_origin + 0.16 * dir_0, color=BLACK, stroke_width=3
        )
        label_0 = MathTex("0", color=BLACK).move_to(
            p_amp_origin + 0.65 * dir_0 + np.array([0, 0, -0.05])
        )

        # Znacznik szwu pi / -pi (dokladnie naprzeciwko po prawej stronie)
        p_pi = np.array([R_axis * np.cos(alpha_pi), R_axis * np.sin(alpha_pi), Z_base])
        dir_pi = np.array([np.cos(alpha_pi), np.sin(alpha_pi), 0])
        tick_pi = Line(p_pi - 0.16 * dir_pi, p_pi + 0.16 * dir_pi, color=BLACK, stroke_width=3)
        label_pi = MathTex(r"\pi / -\pi", color=BLACK).move_to(
            p_pi + 1.65 * dir_pi + np.array([0, 0, -0.05])
        )

        # Etykieta osi fazy na dole z przodu pod grotem
        dir_label = np.array([np.cos(u_phase_tip), np.sin(u_phase_tip), 0])
        label_phase = Tex("Phase", color=BLACK).move_to(
            p_phase_tip + 0.70 * dir_label + np.array([0, 0, -0.45])
        )

        # --- Przeplyw (Flow trajectory): x_0 -> x_1 przez szew na tyl walca ---
        angle_start = alpha_pi - 0.40
        angle_end = alpha_pi + 1.30
        z_start = 1.5
        z_end = -1.5

        p_start = np.array([R_curve * np.cos(angle_start), R_curve * np.sin(angle_start), z_start])
        p_end = np.array([R_curve * np.cos(angle_end), R_curve * np.sin(angle_end), z_end])

        dot_start = Dot3D(p_start, color=BLUE, radius=0.11)
        dot_end = Dot3D(p_end, color=BLUE, radius=0.08)

        # Wektor predkosci wzdluz trajektorii i uklad w plaszczyznie stycznej walca w p_end
        dtheta = angle_end - angle_start
        dz = z_end - z_start
        v_end = np.array(
            [-R_curve * np.sin(angle_end) * dtheta, R_curve * np.cos(angle_end) * dtheta, dz]
        )
        v_len = np.linalg.norm(v_end)
        u_tangent = v_end / v_len
        n_surface = np.array([np.cos(angle_end), np.sin(angle_end), 0])
        w_perp = np.cross(u_tangent, n_surface)
        w_perp = w_perp / np.linalg.norm(w_perp)

        # Mniejszy grot 2D lezacy scisle w plaszczyznie stycznej walca przy x_1
        L_arrow = 0.22
        W_arrow = 0.10
        p_arrow_tip = p_end - 0.04 * u_tangent
        p_arrow_base = p_arrow_tip - L_arrow * u_tangent

        arrow_head = Polygon(
            p_arrow_tip,
            p_arrow_base + W_arrow * w_perp,
            p_arrow_base - W_arrow * w_perp,
            color=RED,
            fill_color=RED,
            fill_opacity=1.0,
            stroke_width=0,
        )

        # Krzywa trajektorii z zwezajaca sie gruboscia (tapering: 7 -> 4)
        t_max = 1.0 - ((0.04 + 0.85 * L_arrow) / v_len)
        n_segs = 80
        flow_segments = VGroup()
        for i in range(n_segs):
            t0 = (i / n_segs) * t_max
            t1 = ((i + 1) / n_segs) * t_max
            pt0 = np.array(
                [
                    R_curve * np.cos(angle_start * (1 - t0) + angle_end * t0),
                    R_curve * np.sin(angle_start * (1 - t0) + angle_end * t0),
                    z_start * (1 - t0) + z_end * t0,
                ]
            )
            pt1 = np.array(
                [
                    R_curve * np.cos(angle_start * (1 - t1) + angle_end * t1),
                    R_curve * np.sin(angle_start * (1 - t1) + angle_end * t1),
                    z_start * (1 - t1) + z_end * t1,
                ]
            )
            sw = 7.0 - 3.0 * (i / (n_segs - 1))
            flow_segments.add(Line(pt0, pt1, color=RED, stroke_width=sw))

        # Etykiety punktow
        label_x0 = MathTex("x_0", color=BLACK).move_to(p_start + np.array([0, 0, 0.38]))
        label_x1 = MathTex("x_1", color=BLACK).move_to(p_end + np.array([0, 0, -0.38]))

        self.add(
            cylinder,
            seam_line,
            amp_axis,
            tick_0,
            phase_circle,
            phase_tip,
            tick_pi,
            flow_segments,
            arrow_head,
            dot_start,
            dot_end,
        )
        self.add_fixed_orientation_mobjects(
            label_amp, label_0, label_pi, label_phase, label_x0, label_x1
        )
