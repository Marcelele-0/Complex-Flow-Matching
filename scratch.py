with open("tests/test_eval/test_reconstruct.py") as f:
    content = f.read()

content = content.replace(
    "from cfm.flow.bridge import GeodesicFlowBridge",
    "from cfm.manifolds.cylindrical import CylindricalManifold",
)
content = content.replace("from cfm.flow.solver import CylindricalODESolver\n", "")

body_old = """    def test_reconstruct_slice_shape_and_dtype(self) -> None:
        x_1 = _synthetic_cylindrical_slice(16, 16)
        solver = CylindricalODESolver(num_steps=3)
        bridge = GeodesicFlowBridge()

        out, x_corr = reconstruct_slice(
            model=_dummy_model,
            solver=solver,
            bridge=bridge,
            x_1=x_1,
            t_start=0.5,
        )"""

body_new = """    def test_reconstruct_slice_shape_and_dtype(self) -> None:
        x_1 = _synthetic_cylindrical_slice(16, 16)
        manifold = CylindricalManifold()
        solver = manifold.make_solver(num_steps=3)

        out, x_corr = reconstruct_slice(
            model=_dummy_model,
            manifold=manifold,
            solver=solver,
            x_1=x_1,
            t_start=0.5,
        )"""
content = content.replace(body_old, body_new)

with open("tests/test_eval/test_reconstruct.py", "w") as f:
    f.write(content)
