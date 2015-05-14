<!DOCTYPE html>

<html>

    <head>

        <link href="/css/bootstrap.min.css" rel="stylesheet"/>
        <link href="/css/bootstrap-theme.min.css" rel="stylesheet"/>
        <link href="/css/style.css" rel="stylesheet"/>

        <?php if (isset($title)): ?>
            <title><?= htmlspecialchars($title) ?></title>
        <?php else: ?>
            <title>Around me</title>
        <?php endif ?>

        
    </head>

    