import 'package:braindance/configs/app_config.dart';
import 'package:flutter/material.dart';

class RecallRenameModelDialog extends StatefulWidget {
  final String initialName;

  const RecallRenameModelDialog({super.key, required this.initialName});

  @override
  State<RecallRenameModelDialog> createState() =>
      _RecallRenameModelDialogState();
}

class _RecallRenameModelDialogState extends State<RecallRenameModelDialog> {
  static final _invalidChars = RegExp(r'[/\\:*?"<>|]');
  late final TextEditingController _controller;
  String? _errorText;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: widget.initialName);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: Text(textLocalize('recall_rename_model')),
      content: TextField(
        controller: _controller,
        autofocus: true,
        decoration: InputDecoration(
          hintText: textLocalize('recall_rename_hint'),
          errorText: _errorText,
        ),
        onChanged: (value) {
          setState(() {
            _errorText = _invalidChars.hasMatch(value)
                ? textLocalize('recall_rename_invalid')
                : null;
          });
        },
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: Text(textLocalize('gen_cancel')),
        ),
        TextButton(
          onPressed: () {
            final text = _controller.text.trim();
            if (text.isEmpty || _invalidChars.hasMatch(text)) {
              return;
            }
            Navigator.pop(context, text);
          },
          child: Text(textLocalize('gen_button')),
        ),
      ],
    );
  }
}
