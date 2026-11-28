import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from modules_forge.native_recipes import ideogram_recipe, ltx_recipe, seed_sequence, native_gallery, atomic_json, read_native_metadata, native_infotext
from modules_forge.pruna_decoder import decoder_widths, pruna_construction
from modules.grouped_inputs import source_indices, group_sources


class NativeRemainingTests(unittest.TestCase):
    def test_named_ideogram_recipes(self):
        for name, steps, tail, mu, std in (("Quality 48", 48, 3, 0, 1.5), ("Default 20", 20, 2, 0, 1.75), ("Turbo 12", 12, 1, .5, 1.75)):
            r = ideogram_recipe(name, 1, 2, 3, 4, False)
            self.assertEqual((r["num_inference_steps"], r["mu"], r["std"]), (steps, mu, std))
            self.assertEqual(r["guidance_schedule"], [7.] * (steps-tail) + [3.] * tail)
            self.assertIsNone(r["guidance_scale"])

    def test_custom_schedule(self):
        r = ideogram_recipe("Custom", 3, 7, 0, 1.5, True, "7,5,0")
        self.assertEqual(r["guidance_schedule"], [7, 5, 0])
        for s in ("7,3", "7,nan,3", "7,-1,3", "7,25,3"):
            with self.assertRaises(ValueError):
                ideogram_recipe("Custom", 3, 7, 0, 1.5, True, s)

    def test_two_stage_recipe_does_not_mutate_input(self):
        args = dict(width=768, height=512, guidance_scale=3, output_type="np", num_inference_steps=30, stg_scale=1, audio_stg_scale=1)
        first, second = ltx_recipe("Distilled + refine 2x", args, [1,.5], [.9,.4])
        self.assertEqual((first["num_inference_steps"], first["guidance_scale"], first["stg_scale"], first["audio_stg_scale"]), (2,1,0,0))
        self.assertEqual((second["width"], second["height"], second["noise_scale"], second["output_type"]), (1536,1024,.9,"np"))
        self.assertEqual(args["output_type"], "np")
        self.assertIsNone(ltx_recipe("Full", args, [1], [.9])[1])
        with self.assertRaises(ValueError):
            ltx_recipe("bad", args, [1], [.9])

    def test_seed_and_gallery(self):
        seeds = seed_sequence(2**63-1, 3)
        self.assertEqual(seeds, [2**63-1,0,1])
        md = dict(prompt="sample", negative_prompt="", request_seed=seeds[0], width=32, height=32, steps=12, cfg_scale=7, sampler_name="Ideogram 4", sd_model_name="local")
        texts = [native_infotext(md,s) for s in seeds]
        g = native_gallery(md, seeds, texts)
        self.assertEqual(g["all_seeds"], seeds)
        self.assertEqual(g["all_prompts"], ["sample"]*3)
        self.assertEqual(g["infotexts"], texts)
        self.assertEqual(read_native_metadata(SimpleNamespace(info={"forge_native":json.dumps(md)})),md)
        with self.assertRaises(ValueError):
            native_gallery(md, seeds, texts[:1])

    def test_atomic_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"clip.json"
            atomic_json(p,{"seed":1})
            with self.assertRaises(ValueError):
                atomic_json(p,{"bad":float("nan")})
            self.assertEqual(json.loads(p.read_text()),{"seed":1})
            self.assertEqual(list(Path(root).iterdir()),[p])

    def test_pruna_widths(self):
        self.assertEqual(decoder_widths((128,256,384,1024),(2,2,1,2)),[(1024,1024,512),(512,384,384),(384,256,128),(128,128,64)])
        with self.assertRaises(ValueError):
            decoder_widths((3,),(2,))

    def test_pruna_scope_restores_constructors(self):
        class Block:
            def __init__(self,in_channels,out_channels=None,num_layers=1,dropout=0.,resnet_eps=1e-6,resnet_act_fn="swish",spatio_temporal_scale=True,upsample_type="spatial",inject_noise=False,timestep_conditioning=False,upsample_residual=False,upscale_factor=1,spatial_padding_mode="zeros"):
                self.conv_in = None if in_channels == out_channels else (in_channels,out_channels)
        class Decoder:
            def __init__(self, block_out_channels, upsample_factor):
                self.blocks = [Block(n//f,n//f,upscale_factor=f) for n,f in zip(reversed(block_out_channels),reversed(upsample_factor))]
        module = SimpleNamespace(LTX2VideoDecoder3d=Decoder,LTX2VideoUpBlock3d=Block,LTX2VideoResnetBlock3d=lambda **kw:(kw["in_channels"],kw["out_channels"]))
        original = (Decoder.__init__, Block.__init__)
        cfg = {"decoder_block_out_channels":[128,256,384,1024],"upsample_factor":[2,2,1,2]}
        with pruna_construction(module,"0.39.0",cfg):
            result = Decoder(cfg["decoder_block_out_channels"],cfg["upsample_factor"])
            self.assertEqual([b.conv_in for b in result.blocks],[None,(512,384),(384,256),None])
        self.assertEqual((Decoder.__init__, Block.__init__),original)
        with self.assertRaises(RuntimeError):
            with pruna_construction(module,"0.40.0",cfg):
                pass
        self.assertEqual((Decoder.__init__, Block.__init__),original)

    def test_source_associations(self):
        self.assertEqual(source_indices([0,2,4],3,3),[0,2,1])
        self.assertEqual(source_indices([2,4],3,1),[0,0])
        p = SimpleNamespace(init_images=["a","b","c"])
        with self.assertRaises(RuntimeError):
            with group_sources(p,p.init_images,3,[4,0]):
                self.assertEqual(p.init_images,["b","a"])
                raise RuntimeError()
        self.assertEqual(p.init_images,["a","b","c"])
        with self.assertRaises(ValueError):
            source_indices([0],3,2)


if __name__ == "__main__":
    unittest.main()
