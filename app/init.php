<?php
namespace app;
use library\utils;

class init{
   
    
    public function exceptionHandler(){
        set_exception_handler(array($this,"missedException"));
    }
    public static function missedException($e){
        echo "Sorry, something went wrong. Please try again or contact us if the problem persists.";
        utils::log("Unhandled exception: ".$e->getMessage().", file ".$e->getFile().", line ".$e->getLine());    
    }
    
}

 
