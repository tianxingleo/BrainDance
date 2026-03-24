import 'dart:io';
import 'package:image/image.dart' as img;

void main() {
  final file = File('assets/icon.png');
  if (!file.existsSync()) return;
  final image = img.decodeImage(file.readAsBytesSync());
  if (image == null) return;

  stdout.writeln('Original size: ${image.width}x${image.height}');

  int transparent = 0, opaque = 0;
  for (int y = 0; y < image.height; y += 5) {
    for (int x = 0; x < image.width; x += 5) {
      if (image.getPixel(x, y).a < 128) {
        transparent++;
      } else {
        opaque++;
      }
    }
  }
  stdout.writeln('transparent pixels: $transparent, opaque pixels: $opaque');

  final w = image.width, h = image.height;
  final size = w > h ? w : h;
  final dstX = (size - w) ~/ 2;
  final dstY = (size - h) ~/ 2;

  // Transparent background version
  final trans = img.Image(width: size, height: size);
  img.fill(trans, color: img.ColorRgba8(0, 0, 0, 0));
  img.compositeImage(trans, image, dstX: dstX, dstY: dstY);
  File('assets/icon_square_transparent.png').writeAsBytesSync(img.encodePng(trans));

  // White background version (splash light mode)
  final light = img.Image(width: size, height: size);
  img.fill(light, color: img.ColorRgba8(255, 255, 255, 255));
  img.compositeImage(light, image, dstX: dstX, dstY: dstY);
  File('assets/icon_square_opaque.png').writeAsBytesSync(img.encodePng(light));

  // Dark background version (splash dark mode)
  final dark = img.Image(width: size, height: size);
  img.fill(dark, color: img.ColorRgba8(18, 18, 18, 255));
  img.compositeImage(dark, image, dstX: dstX, dstY: dstY);
  File('assets/icon_square_dark.png').writeAsBytesSync(img.encodePng(dark));

  stdout.writeln(
    'Done. Transparent: ${size}x$size, Opaque white: ${size}x$size, Opaque dark: ${size}x$size',
  );
}
