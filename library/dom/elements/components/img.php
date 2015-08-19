<?php
/**
 * Description of img
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\void;

class img extends void{

    
    public function __construct($src=false) {
        parent::__construct();
        $this->tag = "img";
        if($src){
            $this->attributes["src"] = $src;
        }
    }


}
