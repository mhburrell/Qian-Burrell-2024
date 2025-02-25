# run_td_sims.ps1
# Usage: .\run_td_sims.ps1


# Array of alpha values
$alphaValues = 0.1

For ($rep = 1; $rep -le 25; $rep++) {
    For ($model_switch = 1; $model_switch -le 3; $model_switch++) {
        For ($testgroup = 1; $testgroup -le 3; $testgroup++) {

            # If you prefer an integer loop to avoid floating-point accumulation:
            # For ($i = 0; $i -le 19; $i++) {
            #     $gamma = 0.5 + (0.025 * $i)

            For ($gamma = 0.5; $gamma -le 0.975; $gamma += 0.025) {
                # Round gamma a bit if you want to avoid floating-point artifacts:
                $gammaRounded = [Math]::Round($gamma, 4)

                foreach ($alpha in $alphaValues) {
                    Write-Host "Running: python do_td_sim_params.py $rep $model_switch $testgroup $gammaRounded $alpha"
                    python do_td_sim_params.py $rep $model_switch $testgroup $gammaRounded $alpha
                }
            }
        }
    }
}
