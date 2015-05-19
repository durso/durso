<?php

/**
 * Description of response
 *
 * @author durso
 */
namespace app;

class response {
    public static function flush(){
        ob_flush();
    }
    public static function endFlush() {
        ob_end_flush();
    }
    public static function clearBuffer(){
        ob_end_clean();
    }
}
