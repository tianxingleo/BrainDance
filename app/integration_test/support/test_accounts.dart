class TestAccount {
  const TestAccount({
    required this.email,
    required this.password,
    required this.label,
  });

  final String email;
  final String password;
  final String label;
}

class TestAccounts {
  static const userA = TestAccount(
    email: 'user_a@test.local',
    password: 'BrainDance-It-UserA-2026',
    label: 'user_a',
  );

  static const userB = TestAccount(
    email: 'user_b@test.local',
    password: 'BrainDance-It-UserB-2026',
    label: 'user_b',
  );

  static const admin = TestAccount(
    email: 'admin@test.local',
    password: 'BrainDance-It-Admin-2026',
    label: 'admin',
  );
}
