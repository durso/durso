<?php
/**
 * Description of formBusca
 *
 * @author durso
 */
namespace library\layout\components;
use library\layout\components\component;

 

class formBusca extends component{
    
    public function __construct($rows,$tag = "div",$class = array("row")) {
        $path = VIEW_PATH.DS."templates/formbusca.php"; 
        $this->value = include($path);
        $this->tag = $tag;
        $this->closeTag = true;
        $this->attributes["class"] = $class;
    }
   
    

    
}