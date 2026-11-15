"""Check portable image previews without real catalog files or ML dependencies."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.preview_model_comparison import build_preview


class ModelPreviewTests(unittest.TestCase):
    def bundle(self, root, key='top/anchor.jpg'):
        data={'schema_version':1,'training_seed':42,'sample_seed':2026,
              'pool_sizes':{'pants':1,'shoes':1},'models':[],
              'items':{'anchor</script>':{'image_key':key,'category':'top'}},
              'queries':[{'anchor':'anchor</script>','partners':{'pants':[],'shoes':[]},'predictions':{}}]}
        path=root/'predictions.json';path.write_text(json.dumps(data));return path

    def test_embeds_local_images_and_escapes_script_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);images=root/'images';(images/'top').mkdir(parents=True)
            Image.new('RGB',(20,20),'red').save(images/'top/anchor.jpg')
            output=root/'preview.html'
            data=build_preview(self.bundle(root),images,output)
            self.assertEqual(data['missing_images'],0)
            self.assertTrue(data['thumbnails']['anchor</script>'].startswith('data:image/jpeg;base64,'))
            html=output.read_text()
            self.assertEqual(html.count('</script>'),1)
            payload=html.split('const data=',1)[1].split(';\nconst byId=',1)[0]
            self.assertIn('anchor</script>',json.loads(payload)['items'])

    def test_missing_images_require_explicit_placeholder_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=self.bundle(root)
            with self.assertRaisesRegex(ValueError,'No images'):
                build_preview(bundle,root/'images',root/'preview.html')
            data=build_preview(bundle,root/'images',root/'preview.html',allow_missing=True)
            self.assertEqual(data['missing_images'],1)

    def test_image_keys_cannot_escape_image_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(ValueError,'escapes'):
                build_preview(self.bundle(root,'../outside.jpg'),root/'images',root/'preview.html')


if __name__=='__main__':unittest.main()
