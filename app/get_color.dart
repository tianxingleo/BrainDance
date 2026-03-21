import 'dart:io';
import 'package:image/image.dart' as img;

void main() {
  final file = File('assets/icon.png');
  if (!file.existsSync()) return;
  final image = img.decodeImage(file.readAsBytesSync());
  if (image == null) return;

  final bgColor = image.getPixel(0, 0);
  stdout.writeln(
    'Top-left: r=${bgColor.r}, g=${bgColor.g}, b=${bgColor.b}, a=${bgColor.a}',
  );
  
  // Let's sample the center of the top edge instead, or just a bit inside
  final centerTop = image.getPixel(image.width ~/ 2, 10);
  stdout.writeln(
    'Center-top: r=${centerTop.r}, g=${centerTop.g}, b=${centerTop.b}, a=${centerTop.a}',
  );
}
