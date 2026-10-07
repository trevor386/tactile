"""MJCF export: structure (no MuJoCo needed) and equivalence with the description (needs ``mujoco``)."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch

from somato.geometry.rotations import matrix_to_quat, quat_to_matrix, random_rotation, rpy_to_matrix
from somato.robots.description import Body, Joint, RobotDescription, Shape
from somato.robots.factory import build_robot
from somato.robots.mjcf import description_to_mjcf, write_mjcf

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def snake16():
    desc, _ = build_robot(ROOT / "configs" / "robots" / "snake_3d.yaml")
    return desc


@pytest.fixture(scope="module")
def zoo():
    """Branching robot with every joint/shape kind, off-axis frames and non-axis-aligned shapes."""
    bodies = [
        Body("base", 2.0, (0.011, 0.013, 0.017), (0.01, -0.02, 0.03),
             [Shape("box", {"x": 0.3, "y": 0.2, "z": 0.1}, (0.0, 0.0, 0.01), (0.1, -0.2, 0.3)),
              Shape("sphere", {"radius": 0.05}, (0.1, 0.0, 0.0))]),
        Body("arm", 0.7, (0.004, 0.002, 0.003), (0.0, 0.0, 0.1),
             [Shape("cylinder", {"radius": 0.02, "length": 0.15}, (0.0, 0.0, 0.1), (0.3, 0.0, -0.4))]),
        Body("slider", 0.3, (0.001, 0.002, 0.0015), (0.01, 0.0, 0.0),
             [Shape("capsule", {"radius": 0.01, "length": 0.08}, (0.04, 0.0, 0.0), (0.0, math.pi / 2, 0.0))]),
        Body("wheel", 0.4, (0.002, 0.002, 0.0035), (0.0, 0.0, 0.0),
             [Shape("cylinder", {"radius": 0.06, "length": 0.03}, (0.0, 0.0, 0.0), (math.pi / 2, 0.0, 0.0))]),
        Body("fixed_tip", 0.1, (1e-4, 2e-4, 2.5e-4), (0.0, 0.0, 0.02), [Shape("sphere", {"radius": 0.02})]),
        Body("leaf", 0.2, (1e-3, 1e-3, 1e-3)),  # no shapes
    ]
    joints = [
        Joint("j_arm", "revolute", "base", "arm", (0.1, 0.05, -0.02), (0.2, -0.5, 1.1), (1.0, 2.0, -0.5),
              lower=-1.2, upper=0.9, effort=12.0),
        Joint("j_slide", "prismatic", "arm", "slider", (0.0, 0.0, 0.2), (0.0, 0.7, 0.0), (0.0, 0.6, 0.8),
              lower=-0.05, upper=0.07, effort=30.0),
        Joint("j_wheel", "continuous", "base", "wheel", (-0.1, 0.2, 0.0), (-0.3, 0.1, 0.2), (0.0, 1.0, 0.0),
              effort=5.0),
        Joint("j_tip", "fixed", "arm", "fixed_tip", (0.0, 0.0, 0.3), (0.5, 0.0, 0.0)),
        Joint("j_leaf", "revolute", "fixed_tip", "leaf", (0.0, 0.1, 0.0), (0.0, 0.0, 0.4), (1.0, 0.0, 0.0),
              lower=-0.4, upper=0.4, effort=2.0),
    ]
    return RobotDescription("zoo", bodies, joints)


# ------------------------------------------------------------------ structure (no mujoco)
def _parse(desc, **kwargs):
    root = ET.fromstring(description_to_mjcf(desc, **kwargs))
    bodies = {b.get("name"): b for b in root.iter("body")}
    return root, bodies


def test_mjcf_is_wellformed_with_header(small_desc):
    root = ET.fromstring(description_to_mjcf(small_desc))
    assert root.tag == "mujoco" and root.get("model") == small_desc.name
    comp = root.find("compiler")
    assert comp.get("angle") == "radian" and comp.get("autolimits") == "true"
    assert comp.get("inertiafromgeom") == "false"
    assert [c.tag for c in root] == ["compiler", "worldbody"]  # no option/actuator/sensor/lights/ground


def test_names_and_counts(snake16):
    root, bodies = _parse(snake16)
    assert list(bodies) == snake16.body_names
    joints = [j.get("name") for j in root.iter("joint")]
    assert joints == snake16.joint_names
    assert [f.get("name") for f in root.iter("freejoint")] == ["root"]
    geoms = [g.get("name") for g in root.iter("geom")]
    assert geoms == [f"{n}_geom0" for n in snake16.body_names]


def test_tree_nesting_and_root_freejoint(zoo):
    root, bodies = _parse(zoo)
    (top,) = root.find("worldbody")
    assert top.get("name") == zoo.root_body
    assert top[0].tag == "freejoint"  # first child of the root body
    for j in zoo.joints:
        assert j.child in [c.get("name") for c in bodies[j.parent].findall("body")]
    # fixed joint: child welded, no <joint> element; continuous: unlimited
    assert bodies["fixed_tip"].find("joint") is None
    wheel = bodies["wheel"].find("joint")
    assert wheel.get("type") == "hinge" and wheel.get("limited") == "false" and wheel.get("range") is None
    slide = bodies["slider"].find("joint")
    assert slide.get("type") == "slide" and slide.get("range") is not None
    assert bodies["arm"].find("joint").get("type") == "hinge"


def test_fixed_base_has_no_freejoint(small_desc):
    root, _ = _parse(small_desc, floating_base=False)
    assert root.find(".//freejoint") is None


def test_body_and_geom_frames_in_xml(zoo):
    root, bodies = _parse(zoo)
    for j in zoo.joints:
        el = bodies[j.child]
        assert [float(v) for v in el.get("pos").split()] == pytest.approx(j.pos, abs=1e-8)
        R = quat_to_matrix(torch.tensor([float(v) for v in el.get("quat").split()], dtype=torch.float64))
        assert torch.allclose(R, rpy_to_matrix(torch.tensor(j.rpy, dtype=torch.float64)), atol=1e-8)
    for b in zoo.bodies:
        inertial = bodies[b.name].find("inertial")
        assert float(inertial.get("mass")) == pytest.approx(b.mass)
        assert [float(v) for v in inertial.get("pos").split()] == pytest.approx(b.com, abs=1e-8)
        assert [float(v) for v in inertial.get("diaginertia").split()] == pytest.approx(b.inertia, rel=1e-7)
        geoms = bodies[b.name].findall("geom")
        assert [g.get("name") for g in geoms] == [f"{b.name}_geom{i}" for i in range(len(b.shapes))]


def test_geom_types_and_sizes(zoo):
    _, bodies = _parse(zoo)
    g = {b.name: bodies[b.name].findall("geom") for b in zoo.bodies}
    # cylinder/capsule -> capsule(radius, half cylindrical length)
    sizes = {n: [float(v) for v in g[n][0].get("size").split()] for n in ("arm", "slider", "wheel")}
    assert sizes["arm"] == pytest.approx([0.02, 0.075]) and g["arm"][0].get("type") == "capsule"
    assert sizes["slider"] == pytest.approx([0.01, 0.04]) and g["slider"][0].get("type") == "capsule"
    assert sizes["wheel"] == pytest.approx([0.06, 0.015]) and g["wheel"][0].get("type") == "capsule"
    assert g["base"][0].get("type") == "box"
    assert [float(v) for v in g["base"][0].get("size").split()] == pytest.approx([0.15, 0.1, 0.05])
    assert g["base"][1].get("type") == "sphere" and float(g["base"][1].get("size")) == pytest.approx(0.05)


def test_snake_capsule_size(small_desc):
    _, bodies = _parse(small_desc)
    shape = small_desc.bodies[0].shapes[0]
    for b in small_desc.bodies:
        (g,) = bodies[b.name].findall("geom")
        assert g.get("type") == "capsule"
        size = [float(v) for v in g.get("size").split()]
        assert size == pytest.approx([shape.size["radius"], shape.size["length"] / 2])


def test_cylinders_as_capsules_flag(zoo):
    _, bodies = _parse(zoo, cylinders_as_capsules=False)
    assert bodies["arm"].find("geom").get("type") == "cylinder"  # cylinder shape
    assert bodies["wheel"].find("geom").get("type") == "cylinder"
    assert bodies["slider"].find("geom").get("type") == "capsule"  # capsule shapes stay capsules


def test_self_collision_flag_in_xml(small_desc):
    root, _ = _parse(small_desc, self_collision=False)
    assert all(g.get("contype") == "0" and g.get("conaffinity") == "1" for g in root.iter("geom"))
    root, _ = _parse(small_desc, self_collision=True)
    assert all(g.get("contype") is None and g.get("conaffinity") is None for g in root.iter("geom"))


def test_sites_in_xml(small_desc):
    sites = [("imu", "link_0", (0.05, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
             ("tip", "link_3", (0.1, 0.01, -0.02), (0.5, 0.5, 0.5, 0.5))]
    _, bodies = _parse(small_desc, sites=sites)
    assert [s.get("name") for s in bodies["link_0"].findall("site")] == ["imu"]
    (tip,) = bodies["link_3"].findall("site")
    assert tip.get("name") == "tip" and tip.get("size") == "0.005"
    assert [float(v) for v in tip.get("pos").split()] == pytest.approx([0.1, 0.01, -0.02])
    assert [float(v) for v in tip.get("quat").split()] == pytest.approx([0.5] * 4)
    assert sum(len(b.findall("site")) for b in bodies.values()) == 2


def test_invalid_inputs(small_desc):
    with pytest.raises(ValueError, match="unknown body"):
        description_to_mjcf(small_desc, sites=[("s", "nope", (0, 0, 0), (1, 0, 0, 0))])
    bad = RobotDescription("bad", [Body("a", shapes=[Shape("mesh", {})])], [])
    with pytest.raises(ValueError, match="Unsupported shape"):
        description_to_mjcf(bad)


def test_write_and_method(tmp_path, small_desc):
    path = write_mjcf(small_desc, tmp_path / "sub" / "snake.xml", self_collision=True)
    assert path.read_text() == small_desc.to_mjcf(self_collision=True)
    assert small_desc.write_mjcf(tmp_path / "other.xml").read_text() == description_to_mjcf(small_desc)


# --------------------------------------------------------------- equivalence (needs mujoco)
@pytest.fixture(scope="module")
def mj():
    return pytest.importorskip("mujoco")


def _compile(mj, desc, **kwargs):
    return mj.MjModel.from_xml_string(description_to_mjcf(desc, **kwargs))


def _random_state(desc, batch, seed):
    gen = torch.Generator().manual_seed(seed)
    root_pos = torch.randn(batch, 3, generator=gen, dtype=torch.float64)
    root_rot = random_rotation(batch, gen, dtype=torch.float64)
    q = torch.randn(batch, len(desc.joint_names), generator=gen, dtype=torch.float64) * 0.6
    return root_pos, root_rot, q


def _set_qpos(model, data, desc, root_pos, root_rot, q):
    free = model.jnt_qposadr[model.joint("root").id]
    data.qpos[free:free + 3] = root_pos.numpy()
    data.qpos[free + 3:free + 7] = matrix_to_quat(root_rot).numpy()  # wxyz
    for name, value in zip(desc.joint_names, q.tolist()):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value


def _t(v):
    return torch.as_tensor(v, dtype=torch.float64)


@pytest.fixture(params=["snake16", "small", "zoo"])
def any_desc(request, snake16, small_desc, zoo):
    return {"snake16": snake16, "small": small_desc, "zoo": zoo}[request.param]


def test_compiles_with_expected_sizes(mj, any_desc):
    model = _compile(mj, any_desc)
    assert model.nbody == len(any_desc.bodies) + 1  # + world
    assert model.njnt == len(any_desc.joint_names) + 1  # + free joint
    assert model.nq == 7 + len(any_desc.joint_names)
    assert model.ngeom == sum(len(b.shapes) for b in any_desc.bodies)
    for j in any_desc.joints:
        assert model.body_parentid[model.body(j.child).id] == model.body(j.parent).id
    mj.MjSpec.from_string(description_to_mjcf(any_desc)).compile()  # also loadable via MjSpec (mjlab)


def test_fixed_base_compiles(mj, any_desc):
    model = _compile(mj, any_desc, floating_base=False)
    assert model.njnt == len(any_desc.joint_names) and model.nq == len(any_desc.joint_names)


def test_forward_kinematics_matches_description(mj, any_desc):
    model = _compile(mj, any_desc)
    data = mj.MjData(model)
    root_pos, root_rot, q = _random_state(any_desc, batch=6, seed=3)
    exp_pos, exp_rot = any_desc.forward_kinematics(root_pos, root_rot, q)
    ids = [model.body(n).id for n in any_desc.body_names]
    for k in range(root_pos.shape[0]):
        _set_qpos(model, data, any_desc, root_pos[k], root_rot[k], q[k])
        mj.mj_kinematics(model, data)
        pos = torch.from_numpy(data.xpos[ids].copy())
        rot = torch.from_numpy(data.xmat[ids].reshape(-1, 3, 3).copy())
        assert torch.allclose(pos, exp_pos[k], atol=1e-5), (k, (pos - exp_pos[k]).abs().max())
        assert torch.allclose(rot, exp_rot[k], atol=1e-5), (k, (rot - exp_rot[k]).abs().max())


@pytest.mark.parametrize("as_capsules", [True, False])
def test_geom_types_and_sizes_in_model(mj, any_desc, as_capsules):
    model = _compile(mj, any_desc, cylinders_as_capsules=as_capsules)
    types = mj.mjtGeom
    for b in any_desc.bodies:
        for i, s in enumerate(b.shapes):
            gid = model.geom(f"{b.name}_geom{i}").id
            if s.kind in ("cylinder", "capsule"):
                as_cyl = s.kind == "cylinder" and not as_capsules
                assert model.geom_type[gid] == (types.mjGEOM_CYLINDER if as_cyl else types.mjGEOM_CAPSULE)
                expected = [s.size["radius"], s.size["length"] / 2]  # half the cylindrical section
            elif s.kind == "box":
                assert model.geom_type[gid] == types.mjGEOM_BOX
                expected = [s.size["x"] / 2, s.size["y"] / 2, s.size["z"] / 2]
            else:
                assert model.geom_type[gid] == types.mjGEOM_SPHERE
                expected = [s.size["radius"]]
            assert model.geom_size[gid][: len(expected)] == pytest.approx(expected, abs=1e-8)
            assert model.geom_pos[gid] == pytest.approx(s.pos, abs=1e-8)
            assert model.geom_bodyid[gid] == model.body(b.name).id


def test_com_geom_and_site_poses_match_description(mj, zoo):
    gen = torch.Generator().manual_seed(5)
    site_quat = tuple(matrix_to_quat(random_rotation(1, gen, dtype=torch.float64))[0].tolist())
    sites = [("s0", "slider", (0.01, -0.02, 0.03), site_quat),
             ("s1", "base", (0.0, 0.0, 0.1), (1.0, 0.0, 0.0, 0.0))]
    model = _compile(mj, zoo, sites=sites)
    data = mj.MjData(model)
    root_pos, root_rot, q = _random_state(zoo, batch=1, seed=11)
    _set_qpos(model, data, zoo, root_pos[0], root_rot[0], q[0])
    mj.mj_kinematics(model, data)
    body_pos, body_rot = zoo.forward_kinematics(root_pos, root_rot, q)

    def world(body, pos, rot):
        i = zoo.body_names.index(body)
        return body_pos[0, i] + body_rot[0, i] @ pos, body_rot[0, i] @ rot

    def check(world_pos, world_rot, actual_pos, actual_mat):
        assert torch.allclose(torch.from_numpy(actual_pos.copy()), world_pos, atol=1e-5)
        if world_rot is not None:
            assert torch.allclose(torch.from_numpy(actual_mat.reshape(3, 3).copy()), world_rot, atol=1e-5)

    for b in zoo.bodies:
        p, _ = world(b.name, _t(b.com), torch.eye(3, dtype=torch.float64))
        check(p, None, data.xipos[model.body(b.name).id], None)
        for i, s in enumerate(b.shapes):
            p, R = world(b.name, _t(s.pos), rpy_to_matrix(_t(s.rpy)))
            gid = model.geom(f"{b.name}_geom{i}").id
            check(p, R, data.geom_xpos[gid], data.geom_xmat[gid])
    for name, body, pos, quat in sites:
        sid = model.site(name).id
        assert model.body(model.site_bodyid[sid]).name == body
        assert model.site_pos[sid] == pytest.approx(pos, abs=1e-8)
        assert model.site_quat[sid] == pytest.approx(quat, abs=1e-7)
        p, R = world(body, _t(pos), quat_to_matrix(_t(quat)))
        check(p, R, data.site_xpos[sid], data.site_xmat[sid])


def test_mass_and_inertia_match_description(mj, any_desc):
    model = _compile(mj, any_desc)
    root = model.body(any_desc.root_body).id
    assert model.body_mass.sum() == pytest.approx(any_desc.total_mass(), rel=1e-6)
    assert model.body_subtreemass[root] == pytest.approx(any_desc.total_mass(), rel=1e-6)
    for b in any_desc.bodies:
        i = model.body(b.name).id
        assert model.body_mass[i] == pytest.approx(b.mass, rel=1e-6)
        assert model.body_ipos[i] == pytest.approx(b.com, abs=1e-8)
        # principal moments in the body frame axes: no inertial rotation, not re-sorted
        assert model.body_iquat[i] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-9)
        assert model.body_inertia[i] == pytest.approx(b.inertia, rel=1e-6)


def test_joint_types_axes_and_limits_match_description(mj, any_desc):
    model = _compile(mj, any_desc)
    kinds = {"revolute": mj.mjtJoint.mjJNT_HINGE, "continuous": mj.mjtJoint.mjJNT_HINGE,
             "prismatic": mj.mjtJoint.mjJNT_SLIDE}
    assert model.jnt_type[model.joint("root").id] == mj.mjtJoint.mjJNT_FREE
    for j in any_desc.joints:
        if not j.movable:
            assert model.body_jntnum[model.body(j.child).id] == 0
            continue
        jid = model.joint(j.name).id
        assert model.jnt_bodyid[jid] == model.body(j.child).id
        assert model.jnt_type[jid] == kinds[j.type]
        axis = _t(j.axis)
        assert model.jnt_axis[jid] == pytest.approx((axis / axis.norm()).tolist(), abs=1e-8)
        assert model.jnt_pos[jid] == pytest.approx([0.0, 0.0, 0.0])
        if j.type == "continuous":
            assert not model.jnt_limited[jid]
        else:
            assert model.jnt_limited[jid]
            assert model.jnt_range[jid] == pytest.approx([j.lower, j.upper], abs=1e-7)
        assert model.jnt_actfrclimited[jid]
        assert model.jnt_actfrcrange[jid] == pytest.approx([-j.effort, j.effort])


@pytest.mark.parametrize("self_collision", [False, True])
def test_collision_flags_on_geoms(mj, any_desc, self_collision):
    model = _compile(mj, any_desc, self_collision=self_collision)
    if self_collision:
        assert (model.geom_contype == 1).all() and (model.geom_conaffinity == 1).all()  # MuJoCo defaults
    else:
        assert (model.geom_contype == 0).all() and (model.geom_conaffinity == 1).all()


def _contact_pairs(mj, model, data):
    contacts = data.contact[: data.ncon]
    return {frozenset((model.geom(c.geom1).name, model.geom(c.geom2).name)) for c in contacts}


def test_self_collision_flag_changes_contacts(mj):
    # a - b - c in a row, but c sits 0.05 from a: only the non-adjacent pair (a, c) can collide
    bodies = [Body(n, 1.0, (1e-2, 1e-2, 1e-2), shapes=[Shape("sphere", {"radius": 0.1})]) for n in "abc"]
    joints = [Joint("ab", "revolute", "a", "b", (0.5, 0.0, 0.0)),
              Joint("bc", "revolute", "b", "c", (-0.45, 0.0, 0.0))]
    desc = RobotDescription("bent", bodies, joints)
    for flag, expected in ((False, set()), (True, {frozenset(("a_geom0", "c_geom0"))})):
        model = _compile(mj, desc, self_collision=flag)
        data = mj.MjData(model)
        data.qpos[:7] = [0, 0, 1, 1, 0, 0, 0]
        mj.mj_forward(model, data)
        assert _contact_pairs(mj, model, data) == expected  # parent-child pairs are always filtered


def test_robot_still_collides_with_ground_without_self_collision(mj, small_desc):
    xml = description_to_mjcf(small_desc, self_collision=False)
    xml = xml.replace("</worldbody>", '<geom name="floor" type="plane" size="5 5 0.1"/></worldbody>')
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    data.qpos[:7] = [0, 0, 0.02, 1, 0, 0, 0]  # links (radius 0.025) intersect the floor
    mj.mj_forward(model, data)
    pairs = _contact_pairs(mj, model, data)
    assert pairs and all("floor" in p for p in pairs)
