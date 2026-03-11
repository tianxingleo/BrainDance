import 'package:supabase/supabase.dart';
import 'lib/configs/supabase_config.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

void main() async {
  await dotenv.load(fileName: "lib/.env"); // check if app has .env in lib 
  // try basic initialize
  final supabase = SupabaseClient('http://172.28.97.38:54321', '');
  try {
    final response = await supabase.functions.invoke('search-models', body: {'query': 'test'});
    print(response.data);
  } catch(e) {
    print("Error: \$e");
  }
}
