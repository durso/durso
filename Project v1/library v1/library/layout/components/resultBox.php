<?php

/**
 * Description of resultBox
 *
 * @author durso
 */
namespace library\layout\components;
use library\layout\components\component;

 

class resultBox extends component{
    
    public function __construct($result,$tag = "div",$class = array("row")) {
        $path = VIEW_PATH.DS."templates/thumbnail.php"; 
        $rows = $result;
        $this->value = include($path);
        $this->tag = $tag;
        $this->closeTag = true;
        $this->attributes["class"] = $class;
    }
   
    

    
}
